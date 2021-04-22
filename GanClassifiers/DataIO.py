"""
This file handles the input and output of the data.
It also loads the data and prepares it.
"""
import pandas as pd
import numpy as np

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
    # assumes labeled dataset with 1 column called class (labels)
    sampler = NoLabelSampler()
    return len(sampler.dataset.columns)

def writeResultsToFile(results):
    df = pd.DataFrame(
        data=results,
        columns=["anomalyScore",]
    )
    df.to_csv(PREDICTION_OUTPUT_PATH, index=False)


class NoLabelSampler:
    def __init__(self):
        self.dataset = load_training_set()
        if "Class" in self.dataset.columns:
            self.dataset = self.dataset[self.dataset.Class == 0].drop(
                ["Class"], axis=1
            )

    def __call__(self, batchSize):
        return self.dataset.sample(batchSize).to_numpy().astype(np.float64)
