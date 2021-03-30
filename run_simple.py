import sys
import os
import GanClassifiers

errorMsg = """
Usage: run_me.py classical|tfqSimulator|pennylaneSimulator|pennylaneIBMQ train|predict (--optArgs)
Arguments:
    classical: Run the AnoGAN with a classical generator. This is the fastest option.
    tfqSimulator: Run a simulated version of the quantum AnoGAN with Tensorflow Quantum.
    qulacsSimulator: Run a simulated version of the quantum AnoGAN with Pennylane and the Qulacs simulator backend.
    pennylaneIBMQ: Run on the real hardware with Pennylane and its IBM Q backend.

    train: Trains the network. Requires that input-data/trainingData.csv exists.
    predict: Returns the outlier prediction. Requires that input-data/prediction.csv exists
"""

ganBackends = {
    "classical": GanClassifier.ClassicalClassifier,
    "tfqSimulator": GanClassifiers.TfqSimulator,
    "qulacsSimulator": GanClassifiers.PennylaneSimulator,
    "pennylaneIBMQ": GanClassifiers.PennylaneIbmQ,
}


def main():
    assert len(sys.argv) >= 3, errorMsg
    assert sys.argv[1] in ganBackends.keys(), errorMsg
    assert sys.argv[2] in ["train", "predict"], errorMsg

    classifierClass = ganBackends[sys.argv[1]]
    trainingSteps = 1

    if sys.argv[2] == "train":
        qc = classifierClass(n_steps=trainingSteps)
        qc.train()
        qc.save()
    elif sys.argv[2] == "predict":
        qc = classifierClass.loadClassifier()
        print(qc.predict())
    else:
        print(errorMsg)
    return


if __name__ == "__main__":
    main()
