"""This file is the entrypoint for the docker run command.
Here, the parameters are interpreted and all training/evaluation steps are triggered.
"""

import sys
import GanClassifiers.GANomalyNetworks
import logging

from GanClassifiers.Trainer import Trainer, QuantumDecoderTrainer
from GanClassifiers.slimtasq import EnvironmentVariableManager
from GanClassifiers.DataIO import (writeResultsToFile, DataStorage, save_training_hist)
from GanClassifiers.Plotter import Plotter

DEFAULT_ENV_VARIABLES = {
    "method": "quantum",
    "trainOrpredict": "predict",
    "data_filepath": "",
    "trainingSteps": 10,
    "qcType": "SemiClassicalRandom",
    "quantumDepth": 3,
    "batchSize": 16,
    "discriminatorIterations": 5,
    "validationInterval": 2,
    "validationSamples": 100,
    "discTrainingRate": 0.02,
    "genTrainingRate": 0.02,
    "gpWeight": 10.0,
    "num_shots": 100,
    "latentDim": 6,
    "adv_loss_weight": 1,
    "con_loss_weight": 50,
    "enc_loss_weight": 1,
}


ganBackends = {
    "classical": {"networks": GanClassifiers.GANomalyNetworks.ClassicalDenseNetworks,
                  "trainer": GanClassifiers.Trainer.Trainer,
                  "plotter": GanClassifiers.Plotter.Plotter, }
    ,
    "quantum": {"networks": GanClassifiers.GANomalyNetworks.QuantumDecoderNetworks,
                "trainer": GanClassifiers.Trainer.QuantumDecoderTrainer,
                "plotter": GanClassifiers.Plotter.QuantumDecoderPlotter, },
}


def main():
    # create and configure main logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create the logging file handler
    fh = logging.FileHandler("log.log", mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(fh)

    # Create Singleton object for the first time with the default parameters and perform checks
    envMgr = EnvironmentVariableManager(DEFAULT_ENV_VARIABLES)
    assert envMgr["method"] in ganBackends.keys(), "No valid method parameter provided."
    assert envMgr["trainOrpredict"] in ["train", "predict"], "trainOrpredict parameter not train or predict."
    logger.debug("Environment Variables loaded successfully.")

    # Load data
    data_obj = DataStorage(fp=envMgr["data_filepath"])
    logger.debug("Data loaded successfully.")
    classifier = ganBackends[envMgr["method"]]["networks"](data_obj)
    trainer = ganBackends[envMgr["method"]]["trainer"](data_obj, classifier)
    print("The following models will be used:")
    classifier.print_model_summaries()


    if envMgr["trainOrpredict"] == "train":
        train_hist = trainer.train()
        plotter = ganBackends[envMgr["method"]]["plotter"](train_hist, pix_num_one_side=3)
        plotter.plot()
        save_training_hist(train_hist)
    elif envMgr["trainOrpredict"] == "predict":
        classifier.loadClassifier()
        results = trainer.calculateMetrics(validation_or_test="test")
        print(results)
        plotter = ganBackends[envMgr["method"]]["plotter"](results, fp="model", pix_num_one_side=3, validation=False)
        plotter.plot()
        save_training_hist(results, fp="model/test_results.json")
    return


if __name__ == "__main__":
    main()
