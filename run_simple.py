import sys
import os
from code import QuantumClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

errorMsg = """
Usage: run_me.py train|predict
Arguments:
    train: Trains the network. Requires that input-data/trainingData.csv exists.
    predict: Returns the outlier prediction. Requires that input-data/prediction.csv exists
"""


def main():
    assert len(sys.argv) == 2, errorMsg
    if sys.argv[1] == "train":
        qc = QuantumClassifier(n_steps=1200)
        qc.train()
        qc.save()
    elif sys.argv[1] == "predict":
        qc = QuantumClassifier.loadClassifier()
        print(qc.predict())
    else:
        print(errorMsg)
    return


if __name__ == "__main__":
    main()
