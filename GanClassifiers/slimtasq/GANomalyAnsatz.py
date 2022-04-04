import json
import logging
logger = logging.getLogger(__name__ + ".py")

import inspect
import importlib
import tensorflow as tf


def serialize_element(element):
    #Todo: erase?
    """Serialize an element that might not be in the correct format.

    Args:
        element (any type): element to serialize

    Returns:
        dict: result of the serialization
    """
    success = False
    encoder = tasq.serialize.PennylaneJSONEncoder()
    result = {}
    if isinstance(element, (str, int, list)):
        return element
    try:
        result = encoder.default(element)
    except Exception as e:
        try:
            newElement = {"new": element}
            result = encoder.default(element)
            return result["new"]
        except Exception as e:
            print(
                f"Standard serialization does not for '{element}' with the"
                f" following error:"
            )
            print(e)

    # instead get the weights
    try:
        result = {"weights": element.get_weights()}
        return result
    except Exception as e:
        pass
    return result


class GanAnsatz:
    """
    This Ansatz stores the unique network data / characteristics for general
    Gans. This is not compatible with the GANomaly detection. Use
    GANomaly instead.
    """

    def __init__(self, name):
        """Creates the GanAnsatz object

        Args:
            name (str): The name will be used to identify the current object
            among multiple simulations.
        """
        self._name = name
        self.latent_dim = None
        self._performCheck = True
        self.auto_encoder = None
        self.auto_decoder = None
        self.encoder = None
        self.discriminator = None
        self._trainSampler = None
        self._validationSampler = None
        self._testSampler = None
        self.best_mcc = -1.

    @property
    def trainSampler(self):
        return self._trainSampler

    @property
    def validationSampler(self):
        return self._validationSampler

    @property
    def testSampler(self):
        return self._testSampler

    @property
    def auto_encoder(self):
        return self._auto_encoder

    @property
    def auto_decoder(self):
        return self._auto_decoder

    @property
    def encoder(self):
        return self._encoder

    @property
    def discriminator(self):
        return self._discriminator

    @trainSampler.setter
    def trainSampler(self, sampler):
        self._trainSampler = sampler

    @validationSampler.setter
    def validationSampler(self, sampler):
        self._validationSampler = sampler

    @testSampler.setter
    def testSampler(self, sampler):
        self._testSampler = sampler

    @auto_encoder.setter
    def auto_encoder(self, auto_encoder):
        self._auto_encoder = auto_encoder

    @auto_decoder.setter
    def auto_decoder(self, auto_decoder):
        self._auto_decoder = auto_decoder

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder

    @discriminator.setter
    def discriminator(self, discriminator):
        self._discriminator = discriminator

    def checkAnsatz(self):
        """Perform a check if all elements are set in this Ansatz object.
        If this fails the GAN architecture is likely to fail as well in
        training and evaluation.

        Raises:
            RuntimeError: An element is not set.

        Return:
            Nothing, but if no exception occured everything is in order.
        """
        if self._performCheck is False:
            return
        errorMsg = ""
        if self._auto_encoder is None:
            errorMsg += "The auto_encoder has not been set.\n"
        if self._auto_decoder is None:
            errorMsg += "The auto_decoder has not been set.\n"
        if self._encoder is None:
            errorMsg += "The encoder has not been set.\n"
        if self._discriminator is None:
            errorMsg += "The discriminator has not been set.\n"
        if self._trainSampler is None:
            errorMsg += "The trainSampler has not been set.\n"
        if errorMsg:
            raise RuntimeError(
                f"Ansatz object: {self} did not pass sanity check. The"
                f" following errors were found:\n{errorMsg}"
            )
        else:
            logger.debug("All network models have been successfully defined.")

    def save(self, step=-99, MCC=-99):
        """Store the trained weights and parameters."""
        if step == -99 or MCC == -99:
            data = {
                "latent_dim": self.latent_dim,
            }
            self.auto_encoder.save_weights(
                f"model/checkpoint/{self._name}_auto_encoder_weights"
            )
            self.auto_decoder.save_weights(
                f"model/checkpoint/{self._name}_auto_decoder_weights"
            )
            self.encoder.save_weights(
                f"model/checkpoint/{self._name}_encoder_weights"
            )
            self.discriminator.save_weights(
                f"model/checkpoint/{self._name}_discriminator_weights"
            )
            with open(
                f"model/checkpoint/{self._name}_other_parameters", "w"
            ) as json_file:
                json.dump(data, json_file)
        else:
            self.auto_encoder.save_weights(
                f"model/checkpoint/{self._name}_auto_encoder_weights_step_{step}_MCC_{MCC:.2f}"
            )
            self.auto_decoder.save_weights(
                f"model/checkpoint/{self._name}_auto_decoder_weights_{step}_MCC_{MCC:.2f}"
            )
            self.encoder.save_weights(
                f"model/checkpoint/{self._name}_encoder_weights_{step}_MCC_{MCC:.2f}"
            )
            self.discriminator.save_weights(
                f"model/checkpoint/{self._name}_discriminator_weights_{step}_MCC_{MCC:.2f}"
            )

    def get_config(self):
        #todo: erase?
        """Return the parameters needed to create a copy of this object.
        Overridden method from JSONEncoder.

        Returns:
            dict: parameters
        """
        repr_dict = {}
        toSerialize = [
            "_latentVariableSampler",
            "_name",
            "_trainSampler",
            "_discriminator",
            "_generator",
        ]
        for element in toSerialize:
            # as this is specific logic we perform the recursion here manually
            # maybe it is better to wrap the modules in another class later on
            repr_dict.update(
                {element: serialize_element(getattr(self, element))}
            )
        return repr_dict

    def _to_dict(self):
        #todo: erase?
        """Return the serialization of this object.

        Returns:
            dict: serialized form of this object
        """
        repr_dict = {
            "__class__": self.__class__.__name__,
            "__module__": self.__module__,
            "name": self.name,
        }
        repr_dict.update(self.get_config())
        return repr_dict

    @classmethod
    def _from_dict(cls, dct):
        #todo: erase?
        """Create a new object from a serialized dict.

        Args:
            dct (dict): serialized form of a previous object

        Raises:
            ValueError: dct is missing elements or has invalid parameters.

        Returns:
            GanAnsatz: The new de-serialized object.
        """
        name = dct.pop("name")
        toDeserialize = [
            "_discriminator",
            "_generator",
            "_latentVariableSampler",
            "_name",
            "_trainSampler",
        ]

        obj = cls(name)
        for element in toDeserialize:
            try:
                setattr(obj, element, dct[element])
            except Exception as e:
                print(
                    f"WARNING: {name} could not load element {element} "
                    "due to following error:"
                )
                print(e)
                # this allows tasqw to continue if not everything is loaded
                # -> should be removed once everything supports serialization
        obj._performCheck = False
        return obj


class GANomalyAnsatz(GanAnsatz):
    """ This Ansatz stores the unique network data / characteristics for
    anomaly detection in the AnoGan framework. This abstraction allows
    the use of classical TF, quantum TFQ, and quantum Pennylane
    implementations.
    """

    def __init__(self, name):
        """Creates the GanAnsatz object

        Args:
            name (str): The name will be used to identify the current object
            among multiple simulations.
        """
        super().__init__(name)
        self._GANomalyModel = None
        self._GANomalyInputs = None
        self._discriminatorWeight = 1

    @property
    def GANomaly(self):
        return self._GANomalyModel

    @property
    def discriminatorWeight(self):
        return self._discriminatorWeight

    @discriminatorWeight.setter
    def discriminatorWeight(self, weight):
        self._discriminatorWeight = weight

    @GANomaly.setter
    def GANomalyModel(self, network):
        self._GANomalyModel = network

    @property
    def GANomalyInputs(self):
        # todo make user supplied
        return self._GANomalyInputs

    @GANomalyInputs.setter
    def GANomalyInputs(self, element):
        self._GANomalyInputs = element


    def checkAnsatz(self):
        """Perform a check if all elements are set in this Ansatz object.
        If this fails the GAN architecture is likely to fail as well in
        training and evaluation.

        Raises:
            RuntimeError: An element is not set.

        Return:
            Nothing, but if no exception occured everything is in order.
        """
        if self._performCheck is False:
            return
        errorMsg = ""
        if self._auto_encoder is None:
            errorMsg += "The auto_encoder has not been set.\n"
        if self._auto_decoder is None:
            errorMsg += "The auto_decoder has not been set.\n"
        if self._encoder is None:
            errorMsg += "The encoder has not been set.\n"
        if self._discriminator is None:
            errorMsg += "The discriminator has not been set.\n"
        if self._trainSampler is None:
            errorMsg += "The trainSampler has not been set.\n"
        if self._GANomalyModel is None:
            errorMsg += "The GANomalyModel has not been set.\n"
        if self._GANomalyInputs is None:
            errorMsg += "The GANomalyInputs have not been set.\n"
        if errorMsg:
            raise RuntimeError(
                f"Ansatz object: {self} did not pass sanity check. The "
                f"following errors were found:\n{errorMsg}"
            )

    def get_config(self):
        #todo: erase?
        """Return the parameters needed to create a copy of this object.
        Overridden method from JSONEncoder.

        Returns:
            dict: parameters
        """
        repr_dict = {}
        toSerialize = [
            # "_anoGanInputs",
            "_latentVariableSampler",
            "_name",
            "_trainSampler",
            # "_anoGanModel",
            "_discriminator",
            "_generator",
        ]
        for element in toSerialize:
            # as this is specific logic we perform the recursion here manually
            # maybe it is better to wrap the modules in another class later on
            repr_dict.update(
                {element: serialize_element(getattr(self, element))}
            )
        return repr_dict

    def _to_dict(self):
        #todo: erase?
        """Return the class as a serialized dictionary.

        Returns:
            [dict]: Dictionary representation of the object
        """
        repr_dict = {
            "__class__": self.__class__.__name__,
            "__module__": self.__module__,
            "name": self.name,
        }
        repr_dict.update(self.get_config())
        return repr_dict

    @classmethod
    def _from_dict(cls, dct):
        #todo: erase?
        """Create a new object from a serialized dict.

        Args:
            dct (dict): serialized form of a previous object

        Raises:
            ValueError: dct is missing elements or has invalid parameters.

        Returns:
            GanAnsatz: The new de-serialized object.
        """
        name = dct.pop("name")
        toDeserialize = [
            # "_anoGanInputs",
            # "_anoGanModel",
            "_discriminator",
            "_generator",
            "_latentVariableSampler",
            "_name",
            "_trainSampler",
        ]

        obj = cls(name)
        for element in toDeserialize:
            try:
                setattr(obj, element, dct[element])
            except Exception as e:
                print(
                    f"WARNING: {name} could not load element {element}"
                    " due to following error:"
                )
                print(e)
                # this allows tasq to continue if not everything is loaded
                # -> should be removed once everything supports serialization
        obj._performCheck = False
        return obj
