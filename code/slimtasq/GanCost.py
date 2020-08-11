import numpy as np
import sklearn
import skopt
from .ThresholdWrapper import ThresholdWrapper
import tensorflow as tf
import tensorflow.keras.backend as K
import json


class GanCost:
    def __init__(self, ansatz, **kwargs):
        """Create the cost object.

        Args:
            ansatz (AnoGanAnsatz): This object contains the required structure for the AnoGan ansatz.
        """
        ansatz.checkAnsatz()
        self.ansatz = ansatz
        self.init_params = []

    def calculateMetrics(self, opt):
        """Calculate the metrics. For GANs there are no clear metrics.
        Therefore, this method returns 5 samples from the generator.

        Args:
            opt (tf.keras.optimizer): [unused] Optimizer required for the AnoGan architecture e.g. Adam optimizer.

        Returns:
            dict: dict containing the results for the different metrics.
        """
        inputData = self.ansatz.latentVariableSampler(5)
        snapshots = self.ansatz.generator.predict(inputData)
        return {"samples": snapshots}

    def get_config(self):
        """Return the parameters needed to create a copy of this object.
        Overridden method from JSONEncoder.

        Returns:
            dict: parameters
        """
        config = {"ansatz": self.ansatz._to_dict(), "init_params": self.init_params}
        return config

    def __str__(self):
        return "GanCost " + str(self.ansatz)

    def _to_dict(self):
        """Return the class as a serialized dictionary.

        Returns:
            dict: Dictionary representation of the object
        """
        repr_dict = {
            "__class__": self.__class__.__name__,
            "__module__": self.__module__,
        }
        repr_dict.update(self.get_config())
        return repr_dict

    @classmethod
    def from_config(cls, dct):
        # TensorFlow needs us to deserialize our custom objects first
        ansatz = dct.pop("ansatz")
        ansatz_obj = json.loads(json.dumps(ansatz), cls=PennylaneJSONDecoder)
        dct.update({"ansatz": ansatz_obj, "device": device_obj, "model": model_obj})
        return cls._from_dict(dct)

    @classmethod
    def _from_dict(cls, dct):
        obj = cls(**dct)
        return obj
