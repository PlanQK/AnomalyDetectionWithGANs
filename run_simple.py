#!/usr/bin/python3
"""This file is the entrypoint for the docker run command.
Here, the parameters are interpreted and the respective
AnoGan class/simulation is instanciated.
"""

import sys
import GanClassifiers
from GanClassifiers.slimtasq import EnvironmentVariableManager
from GanClassifiers.DataIO import writeResultsToFile

DEFAULT_ENV_VARIABLES = {
    "method": "classical",
    "trainOrpredict": "train",
    "trainingSteps": 1000,
    "totalDepth": 4,
    "batchSize": 64,
    "discriminatorIterations": 5,
    "adamTrainingRate": 0.01,
    "gpWeight": 1.0,
    "latentVariableOptimizer": "forest_minimize",
    "latentVariableOptimizationIterations": 30,
    "latentDim": 10,
    "ibmqx_token": "",
    "backend": "ibmq_16_melbourne",
}


errorMsg = """
Usage: run_me.py classical|tfqSimulator|pennylaneSimulator|pennylaneIBMQ train|predict (--optArgs)
Arguments:
    classical: Run the AnoGAN with a classical generator. This is the fastest option.
    tfqSimulator: Run a simulated version of the quantum AnoGAN with Tensorflow Quantum.
    pennylaneSimulator: Run a simulated version of the quantum AnoGAN with Pennylane.
    pennylaneIBMQ: Run on the real hardware with Pennylane and its IBM Q backend.

    train: Trains the network. Requires that input-data/trainingData.csv exists.
    predict: Returns the outlier prediction. Requires that input-data/prediction.csv exists

Any further settings are done through environment variables:
    trainingSteps: 1000  Number of iteration for the training of the GAN
    latentDim: 10  size of the latent space = num qubits
    totalDepth: 4  Depth of the circuit or number of layers in the generator
    batchSize: 64  Number of samples per training step
    adamTrainingRate: 0.01  Training rate for the Adam optimizer
    discriminatorIterations: 5  How often does the discriminator update its weights vs generator
    gpWeight: 1.0  Weight factor for the gradient Penalty (Wasserstein Loss specific parameter)
    latentVariableOptimizer: forest_minimize  Which optimizer to choose for the latent variable optimizers
                    possible values: forest_minimize, TF
    latentVariableOptimizationIterations: 30  Number of optimization iterations to obtain the latent variables
    ibmqx_token: ""  Token to access IBM Quantum experience
"""

ganBackends = {
    "classical": GanClassifiers.ClassicalClassifier,
    "tfqSimulator": GanClassifiers.TfqSimulator,
    "pennylaneSimulator": GanClassifiers.PennylaneSimulator,
    "pennylaneIBMQ": GanClassifiers.PennylaneIbmQ,
}


def main():
    # Create Singleton object for the first time with the default parameters
    envMgr = EnvironmentVariableManager(DEFAULT_ENV_VARIABLES)

    assert envMgr["method"] in ganBackends.keys(), errorMsg
    assert envMgr["trainOrpredict"] in ["train", "predict"], errorMsg

    # obtain
    classifierClass = ganBackends[envMgr["method"]]

    if envMgr["trainOrpredict"] == "train":
        qc = classifierClass()
        qc.train()
        qc.save()
        print(f"Number of quantum circuit evaluations: {qc.execution_count_rigetti.execution_counter}"
              f"times repetition number")
    elif envMgr["trainOrpredict"] == "predict":
        qc = classifierClass.loadClassifier()
        results = qc.predict()
        print(results)
        writeResultsToFile(results)

    else:
        print(errorMsg)
    return


if __name__ == "__main__":
    main()
