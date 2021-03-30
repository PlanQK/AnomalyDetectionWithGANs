import inspect
import importlib
import tensorflow as tf


def serialize_element(element):
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
                f"Standard serialization does not for '{element}' with the following error:"
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
    This Ansatz stores the unique network data / characteristics for general Gans.
    This is not compatible with the AnoGan anomaly detection. Use AnoGanAnsatz instead.
    """

    def __init__(self, name):
        """Creates the GanAnsatz object

        Args:
            name (str): The name will be used to identify the current object among multiple simulations.
        """
        self._name = name
        self._performCheck = True
        self._generator = None
        self._discriminator = None
        self._latentVariableSampler = None
        self._trueInputSampler = None

    @property
    def latentVariableSampler(self):
        return self._latentVariableSampler

    @property
    def trueInputSampler(self):
        return self._trueInputSampler

    @property
    def generator(self):
        return self._generator

    @property
    def discriminator(self):
        return self._discriminator

    @latentVariableSampler.setter
    def latentVariableSampler(self, sampler):
        self._latentVariableSampler = sampler

    @trueInputSampler.setter
    def trueInputSampler(self, sampler):
        self._trueInputSampler = sampler

    @generator.setter
    def generator(self, generator):
        self._generator = generator

    @discriminator.setter
    def discriminator(self, discriminator):
        self._discriminator = discriminator

    def checkAnsatz(self):
        """Perform a check if all elements are set in this Ansatz object.
        If this fails the GAN architecture is likely to fail as well in training and in calassification.

        Raises:
            RuntimeError: An element is not set.
        
        Return:
            Nothing, but if no exception occured everything is in order.
        """
        if self._performCheck is False:
            return
        errorMsg = ""
        if self._generator is None:
            errorMsg += "The generator has not been set.\n"
        if self._discriminator is None:
            errorMsg += "The discriminator has not been set.\n"
        if self._latentVariableSampler is None:
            errorMsg += "The latentVariableSampler has not been set.\n"
        if self._trueInputSampler is None:
            errorMsg += "The trueInputSampler has not been set.\n"
        if errorMsg:
            raise RuntimeError(
                f"Ansatz object: {self} did not pass sanity check. The following errors were found:\n{errorMsg}"
            )

    def get_config(self):
        """Return the parameters needed to create a copy of this object.
        Overridden method from JSONEncoder.

        Returns:
            dict: parameters
        """
        repr_dict = {}
        toSerialize = [
            "_latentVariableSampler",
            "_name",
            "_trueInputSampler",
            "_discriminator",
            "_generator",
        ]
        for element in toSerialize:
            # as this is specific logic we perform the recursion here manually
            # maybe it is better to wrap the modules in another class later on
            repr_dict.update({element: serialize_element(getattr(self, element))})
        return repr_dict

    def _to_dict(self):
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
            "_trueInputSampler",
        ]

        obj = cls(name)
        for element in toDeserialize:
            try:
                setattr(obj, element, dct[element])
            except Exception as e:
                print(
                    f"WARNING: {name} could not load element {element} due to following error:"
                )
                print(e)
                # this allows tasqw to continue if not everything is loaded -> should be removed once everything supports serialization
        obj._performCheck = False
        return obj


class AnoGanAnsatz(GanAnsatz):
    """ This Ansatz stores the unique network data / characteristics for anomaly detection in the AnoGan framework.
    This abstraction allows the use of classical TF, quantum TFQ, and quantum Pennylane implementations.
    """

    def __init__(self, name):
        """Creates the GanAnsatz object

        Args:
            name (str): The name will be used to identify the current object among multiple simulations.
        """
        super().__init__(name)
        self._testSampler = None
        self._anoGanModel = None
        self._anoGanInputs = None
        self._trainingDataSampler = None
        self._discriminatorWeight = 1

    @property
    def getTestSample(self):
        return self._testSampler

    @property
    def anoGanModel(self):
        return self._anoGanModel

    @property
    def discriminatorWeight(self):
        return self._discriminatorWeight

    @discriminatorWeight.setter
    def discriminatorWeight(self, weight):
        self._discriminatorWeight = weight

    @anoGanModel.setter
    def anoGanModel(self, network):
        self._anoGanModel = network

    @getTestSample.setter
    def getTestSample(self, sampler):
        self._testSampler = sampler

    @property
    def anoGanInputs(self):
        # todo make user supplied
        return self._anoGanInputs

    @anoGanInputs.setter
    def anoGanInputs(self, element):
        self._anoGanInputs = element

    @property
    def trainingDataSampler(self):
        return self._trainingDataSampler

    @trainingDataSampler.setter
    def trainingDataSampler(self, sampler):
        self._trainingDataSampler = sampler

    def checkAnsatz(self):
        """Perform a check if all elements are set in this Ansatz object.
        If this fails the GAN architecture is likely to fail as well in training and in calassification.

        Raises:
            RuntimeError: An element is not set.
        
        Return:
            Nothing, but if no exception occured everything is in order.
        """
        if self._performCheck is False:
            return
        errorMsg = ""
        if self._generator is None:
            errorMsg += "The generator has not been set.\n"
        if self._discriminator is None:
            errorMsg += "The discriminator has not been set.\n"
        if self._latentVariableSampler is None:
            errorMsg += "The latentVariableSampler has not been set.\n"
        if self._trueInputSampler is None:
            errorMsg += "The trueInputSampler has not been set.\n"
        if self._testSampler is None:
            errorMsg += "The testSampler has not been set.\n"
        if self._anoGanModel is None:
            errorMsg += "The anoGanModel has not been set.\n"
        if self._anoGanInputs is None:
            errorMsg += "The anoGanInputs have not been set.\n"
        if self._trainingDataSampler is None:
            errorMsg += "The trainingDataSampler has not been set.\n"
        if errorMsg:
            raise RuntimeError(
                f"Ansatz object: {self} did not pass sanity check. The following errors were found:\n{errorMsg}"
            )

    def get_config(self):
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
            "_testSampler",
            "_trainingDataSampler",
            "_trueInputSampler",
            # "_anoGanModel",
            "_discriminator",
            "_generator",
        ]
        for element in toSerialize:
            # as this is specific logic we perform the recursion here manually
            # maybe it is better to wrap the modules in another class later on
            repr_dict.update({element: serialize_element(getattr(self, element))})
        return repr_dict

    def _to_dict(self):
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
            "_testSampler",
            "_trainingDataSampler",
            "_trueInputSampler",
        ]

        obj = cls(name)
        for element in toDeserialize:
            try:
                setattr(obj, element, dct[element])
            except Exception as e:
                print(
                    f"WARNING: {name} could not load element {element} due to following error:"
                )
                print(e)
                # this allows tasq to continue if not everything is loaded -> should be removed once everything supports serialization
        obj._performCheck = False
        return obj