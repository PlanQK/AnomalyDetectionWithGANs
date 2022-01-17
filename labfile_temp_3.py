"""
Demonstration file to prototype how rigetti (QVM) can be used for the training of the generator.
"""
from GanClassifiers import GanClassifiers
from GanClassifiers.DataIO import writeResultsToFile
from GanClassifiers.slimtasq import EnvironmentVariableManager

DEFAULT_ENV_VARIABLES = {
    "trainingSteps": 1,
    "totalDepth": 1,
    "batchSize": 1,
    "discriminatorIterations": 5,
    "adamTrainingRate": 0.01,
    "gpWeight": 1.0,
    "latentVariableOptimizer": "forest_minimize",
    "latentVariableOptimizationIterations": 10,
    "latentDim": 2,
    "ibmqx_token": "",
    "backend": "rigetti",
}

def main():
    envMgr = EnvironmentVariableManager(DEFAULT_ENV_VARIABLES)
    qc = GanClassifiers.TfqSimulator()
    qc.train()
    qc.save()
    stop = "stop"


    # qc = GanClassifiers.TfqSimulator.loadClassifier()
    # results = qc.predict()
    # print(results)
    # writeResultsToFile(results)


if __name__ == "__main__":
    main()
