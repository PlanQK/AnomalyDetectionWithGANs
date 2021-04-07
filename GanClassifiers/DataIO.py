"""
This file handles the input and output of the data.
It also loads the data and prepares it.
"""
import pandas as pd
import numpy as np

TRAIN_FILEPATH = "input-data/trainSet.csv"
PREDICTION_FILEPATH = "input-data/predictionSet.csv"


def load_training_set():
    return pd.read_csv(TRAIN_FILEPATH)


def load_prediction_set():
    return pd.read_csv(PREDICTION_FILEPATH)


def load_prediction_set_no_labels():
    data = load_prediction_set()
    try:
        data = data.drop(["Class"], axis=1)
    except Exception:
        pass
    return data


def get_feature_length():
    # assumes labeled dataset with 1 column called class (labels)
    return len(load_training_set().columns) - 1


class NoLabelSampler:
    def __init__(self):
        self.dataset = load_training_set()
        self.dataset = self.dataset[self.dataset.Class == 0].drop(
            ["Class"], axis=1
        )

    def __call__(self, batchSize):
        return self.dataset.sample(batchSize).to_numpy().astype(np.float64)


class LabelSampler:
    def __init__(self):
        self.dataset = load_training_set()

    def __call__(self, batchSize=100):
        dataset = pd.concat(
            [
                self.dataset[self.dataset.Class == 0].sample(
                    int(3 * batchSize / 4), replace=True
                ),
                self.dataset[self.dataset.Class == 1].sample(
                    int(batchSize / 4), replace=True
                ),
            ]
        )
        Y = dataset.Class
        X = dataset.drop(["Class"], axis=1)
        return X, Y
