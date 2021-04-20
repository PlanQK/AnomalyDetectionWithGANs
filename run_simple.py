"""This file is the entrypoint for the docker run command.
Here, the parameters are interpreted and the respective
AnoGan class/simulation is instanciated.
"""

import sys
import GanClassifiers
from GanClassifiers.slimtasq import EnvironmentVariableManager
from GanClassifiers.DataIO import writeResultsToFile
import tensorflow as tf

DEFAULT_ENV_VARIABLES = {
    "trainingSteps": 1000,
    "totalDepth": 4,
    "batchSize": 64,
    "discriminatorIterations": 5,
    "gpWeight": 10,
    "latentVarRandomGuesses": 10,
    "latentVariableOptimizationIterations": 30,
    "latentDim": 10,
    "ibmqx_token": "",
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
    discriminatorIterations: 5  How often does the discriminator update its weights vs Generator
    gpWeight: 10  Weight factor for the gradient Penalty (Wasserstein Loss specific parameter)
    latentVarRandomGuesses: 10  Number of random guesses for the latent variables
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

    assert len(sys.argv) >= 3, errorMsg
    assert sys.argv[1] in ganBackends.keys(), errorMsg
    assert sys.argv[2] in ["train", "predict"], errorMsg

    # obtain
    classifierClass = ganBackends[sys.argv[1]]

    if sys.argv[2] == "train":
        qc = classifierClass()
        qc.train()
        qc.save()
    elif sys.argv[2] == "predict":
        qc = classifierClass.loadClassifier()
        results = qc.predict()
        print(results)
        writeResultsToFile(results)

    else:
        print(errorMsg)
    return


if __name__ == "__main__":
    main()
