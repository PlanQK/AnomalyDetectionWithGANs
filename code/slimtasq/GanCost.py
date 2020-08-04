import numpy as np
import sklearn
import skopt
from .ThresholdWrapper import ThresholdWrapper
import tensorflow as tf
import tensorflow.keras.backend as K
import json


class GanCost:
    def __init__(self, ansatz, **kwargs):
        ansatz.checkAnsatz()
        self.ansatz = ansatz
        self.init_params = []

    def calculateMetrics(self, opt):
        inputData = self.ansatz.latentVariableSampler(5)
        snapshots = self.ansatz.generator.predict(inputData)
        return {"samples": snapshots}

    def get_config(self):
        # Overriding TensorFlow get_config is required if __init__ has additional args
        # For serialization to work with TensorFlow we first serialize Tasq objects via
        # our custom JSONEncoder. Then we convert the JSON string to a dictionary of
        # standard JSON types with json.loads.
        config = {"ansatz": self.ansatz._to_dict(), "init_params": self.init_params}
        return config

    def __str__(self):
        return "GanCost " + str(self.ansatz)

    def _to_dict(self):
        """[1mmary]

        Returns:
            [dict]: Dictionary representation of the class
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
