"""
This file handles the input data.
"""
import os
import pandas as pd
import numpy as np
import json
import logging

from gan_classifiers.EnvironmentVariableManager import EnvironmentVariableManager

logger = logging.getLogger(__name__ + ".py")
class DataStorage:
    """
    Class holding the training, validation and test data. 

    Args:
        fp (string, optional): file path to a target input file.
            Input data requires comma-separated csv files with features 
            + class columns per line.
    """      
    def __init__(self, fp="", print_info=False):
        self.envMgr = EnvironmentVariableManager()
        data_tables = []
        if fp:
            data_tables.append(pd.read_csv(fp))
        else:
            for fp in os.listdir("input_data"):
                data_tables.append(pd.read_csv("input_data/" + fp))
        self.data = pd.concat(data_tables)
        self.feature_length = len(self.data.columns) - 1
        self.normal_samples = self.data[self.data.Class == 0].drop(["Class"], axis=1)
        self.unnormal_samples = self.data[self.data.Class == 1].drop(["Class"], axis=1)
        
        # Divide (Normal, Unnormal) samples for training/validation/test into partitions
        # (80%, 0%)/(10%, 50%)/(10%, 50%)
        partition_normal_indizes = [int(len(self.normal_samples)*0.8), int(len(self.normal_samples)*0.9)]
        partition_unnormal_indizes = int(len(self.unnormal_samples)*0.5)
        
        self.train_data = self.normal_samples[:partition_normal_indizes[0]]
        
        validation_data_normal = self.normal_samples[partition_normal_indizes[0]:partition_normal_indizes[1]]
        validation_data_unnormal = self.unnormal_samples[:partition_unnormal_indizes]
        # Constrain the validation data to the following min value to reach a 50:50 balanced distribution
        # of normal and unnormal samples
        minimum = min(len(validation_data_normal), len(validation_data_unnormal))
        self.validation_data_normal = validation_data_normal[:minimum]
        self.validation_data_unnormal = validation_data_unnormal[:minimum]
        
        test_data_normal = self.normal_samples[partition_normal_indizes[1]:]
        test_data_unnormal = self.unnormal_samples[partition_unnormal_indizes:]
        # Constrain the test data to the following minimum value to reach a 50:50 balanced distribution
        # of normal and unnormal samples
        self.test_data_normal = test_data_normal[:minimum]
        self.test_data_unnormal = test_data_unnormal[:minimum]

        if print_info: # FK: just some logging
            print("Size of training_data: " + str(len(self.train_data)))
            print("Size of validation data: " + str(len(self.validation_data_normal) + len(self.validation_data_unnormal)))
            print("Size of test data: " + str(len(self.test_data_normal) + len(self.test_data_unnormal)))

    def get_train_data(self, batch_size):
        return self.train_data.sample(int(batch_size)).to_numpy().astype(np.float64)

    def get_validation_data(self, batch_size):
        a = self.validation_data_normal.sample(int(batch_size)).to_numpy().astype(np.float64)
        b = self.validation_data_unnormal.sample(int(batch_size)).to_numpy().astype(np.float64)
        return (a, b)

    def get_test_data(self, batch_size=0):
        a = self.test_data_normal.sample(len(self.test_data_normal)).to_numpy().astype(np.float64)
        b = self.test_data_unnormal.sample(len(self.test_data_unnormal)).to_numpy().astype(np.float64)
        return (a, b)

def output_to_json(data, fp = ""):
    """Outputs data to json file """
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    if fp:
        foo = open(fp, "w")
    else:
        # set default file path as backfall case
        foo = open(f"model/data.json", "w")
    json.dump({**data}, foo, cls=NpEncoder)
    foo.close()
    return None