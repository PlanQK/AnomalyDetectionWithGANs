import sys
import os
import numpy as np
import pandas as pd
import sklearn.metrics

def getFalsePositiveRate(Y, prediction):
    """Calculate the false positive rate (FPR).

    Args:
        Y (list): List of actual classes
        prediction (list): List of predictions

    Returns:
        float: false positive rate
    """
    testingsetNegatives = len(Y == 0)
    unique, counts = np.unique(Y.values - prediction, return_counts=True)
    classificationInfo = dict(zip(unique, counts))
    try:
        return float(classificationInfo.get(-1, 0)) / testingsetNegatives
    except:
        return float("nan")

def getFalseNegativeRate(Y, prediction):
    """Calculate th false negative rate (FNR)

    Args:
        Y (list): list of actual results
        prediction (list): list of predictions

    Returns:
        float: false negative rate
    """
    testingsetPositives = len(Y == 1)
    unique, counts = np.unique(Y.values - prediction, return_counts=True)
    classificationInfo = dict(zip(unique, counts))
    try:
        return float(classificationInfo.get(1, 0)) / testingsetPositives
    except:
        return float("nan")

def calcMetrics(labels, predictions):
    return {
        "false positive": getFalsePositiveRate(labels, predictions),
        "false negative": getFalseNegativeRate(labels, predictions),
        "precision": sklearn.metrics.precision_score(labels, predictions),
        "recall": sklearn.metrics.recall_score(labels, predictions),
        "average precision": sklearn.metrics.average_precision_score(labels, predictions),
        "f1 score": sklearn.metrics.f1_score(labels, predictions),
    }

def inferThreshold(anoScoreData, percentageFrauds):
    numExpectedFrauds = int(len(anoScoreData)*percentageFrauds/100)
    assert(1 <= numExpectedFrauds and numExpectedFrauds < len(anoScoreData)-1)
    largestValues = anoScoreData["anomalyScore"].nlargest(numExpectedFrauds+1)[-2:]
    return sum(largestValues)/2

def createPredictions(anoScore, threshold):
    return [
        1 if value >= threshold else 0
        for key, value in anoScore["anomalyScore"].items()
    ]


def main():
    assert len(sys.argv) == 3, "Usage:\ncalculateMetrics.py folder anomalyPercentage"

    anoScore = pd.read_csv(os.path.join(sys.argv[1], "anoScoreResults.csv"))
    labels = pd.read_csv(os.path.join(sys.argv[1], "predictionSet.csv"))["Class"]
    threshold = inferThreshold(anoScore, float(sys.argv[2]))
    predictions = createPredictions(anoScore, threshold)
    print(calcMetrics(labels, predictions))

if __name__ == "__main__":
    main()