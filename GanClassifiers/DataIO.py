"""
This file handles the input and output of the data.
It also loads the data and prepares it.
Data has to consist of comma-separated csv files with features + class columns per line.
"""
import logging

from GanClassifiers.slimtasq import EnvironmentVariableManager

logger = logging.getLogger(__name__ + ".py")

import pandas as pd
import numpy as np
import os
import json

TRAIN_FILEPATH = "model/input-data/trainSet.csv"
PREDICTION_FILEPATH = "model/input-data/predictionSet.csv"
PREDICTION_OUTPUT_PATH = "model/input-data/anoScoreResults.csv"

def load_training_set():
    return pd.read_csv(TRAIN_FILEPATH)

def load_prediction_set():
    return pd.read_csv(PREDICTION_FILEPATH)

def load_prediction_set_no_labels():
    data = load_prediction_set()
    if "Class" in data.columns:
        data = data.drop(["Class"], axis=1)
    return data

def load_prediction_labels():
    return load_prediction_set()["Class"]

def get_feature_length():
    # assumes labeled dataset with 1 column called "Class" (labels)
    sampler = NoLabelSampler()
    return len(sampler.dataset.columns)

def writeResultsToFile(results):
    #Todo: remove or adapt
    df = pd.DataFrame(
        data=results,
        columns=["anomalyScore",]
    )
    df.to_csv(PREDICTION_OUTPUT_PATH, index=False)

class DataStorage:
    """
    Class holding the training and evaluation data. A fp argument can be specified to target one distinct input file.
    """
    def __init__(self, fp=""):
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

    def get_train_data(self, batchSize):
        return self.train_data.sample(int(batchSize)).to_numpy().astype(np.float64)

    def get_validation_data(self, batchSize):
        a = self.validation_data_normal.sample(int(batchSize)).to_numpy().astype(np.float64)
        b = self.validation_data_unnormal.sample(int(batchSize)).to_numpy().astype(np.float64)
        return (a, b)

    def get_test_data(self, batchSize=0):
        a = self.test_data_normal.sample(len(self.test_data_normal)).to_numpy().astype(np.float64)
        b = self.test_data_unnormal.sample(len(self.test_data_unnormal)).to_numpy().astype(np.float64)
        return (a, b)

def save_training_hist(train_hist, fp = ""):

    # Create subclass to enable writing numpy-values to json files
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
        foo = open(f"model/train_history/train_hist.json", "w")
    json.dump({**train_hist}, foo, cls=NpEncoder)
    foo.close()
    return None

class NoLabelSampler:
    """
    Class holding the training data. Training data must only consist of Class=0 samples (normal samples).
    Class column is dropped. Returns "batchSize" samples as a numpy array when called.
    """
    def __init__(self):
        self.dataset = load_training_set()
        if "Class" in self.dataset.columns:
            self.dataset = self.dataset[self.dataset.Class == 0].drop(
                ["Class"], axis=1
            )

    def __call__(self, batchSize):
        return self.dataset.sample(batchSize).to_numpy().astype(np.float64)
