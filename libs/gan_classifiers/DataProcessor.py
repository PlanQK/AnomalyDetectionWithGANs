"""
This file handles the input data.
"""
import pandas as pd
import numpy as np

class Data:
    """
    Class holding the training, validation and test data. 

    Args:
        data : array with data points to be used for the training i.e. training set
        or the classification i.e. test set.
    """      
    def __init__(self, data, parameters):          
        self.data = data
        self.feature_length = len(self.data.columns) - 1

        # Drop the class
        self.normal_samples = self.data[self.data[self.feature_length] == 0].drop(self.feature_length, axis=1)
        self.unnormal_samples = self.data[self.data[self.feature_length] == 1].drop(self.feature_length, axis=1)

        if parameters["train_or_predict"] == "train":          
            # Divide (Normal, Unnormal) samples for training/validation into partitions
            # (80%, 0%)/(20%, 100%)
            partition_normal_indices = [int(len(self.normal_samples)*0.8), int(len(self.normal_samples))]
            partition_unnormal_indices = int(len(self.unnormal_samples))
            
            self.train_data = self.normal_samples[:partition_normal_indices[0]]
            
            validation_data_normal = self.normal_samples[partition_normal_indices[0]:partition_normal_indices[1]]
            validation_data_unnormal = self.unnormal_samples[:partition_unnormal_indices]
            # Constrain the validation data to the following min value to reach a 50:50 balanced distribution
            # of normal and unnormal samples
            minimum = min(len(validation_data_normal), len(validation_data_unnormal))
            self.validation_data_normal = validation_data_normal[:minimum]
            self.validation_data_unnormal = validation_data_unnormal[:minimum]
        else:
            self.test_data_normal = self.normal_samples
            self.test_data_unnormal =  self.unnormal_samples

    def get_train_data(self, batch_size):
        return self.train_data.sample(int(batch_size)).to_numpy().astype(np.float64)

    def get_validation_data(self, batch_size):
        a = self.validation_data_normal.sample(int(batch_size)).to_numpy().astype(np.float64)
        b = self.validation_data_unnormal.sample(int(batch_size)).to_numpy().astype(np.float64)
        return (a, b)

    def get_test_data(self):
        a = self.test_data_normal.to_numpy().astype(np.float64)
        b = self.test_data_unnormal.to_numpy().astype(np.float64)
        return (a, b)