"""This file contains the workflow of the classifier

The parameters of the model are obtained from the environment variables
and the respective training or evaluation steps are triggered.
"""
import logging
import traceback
from typing import Any, Dict, Optional

import pandas

# Import response wrappers:
# - use ResultResponse to return computation results
# - use ErrorResponse to return meaningful error messages to the caller
from libs.return_objects import Response, ResultResponse, ErrorResponse
from libs.utilities import export_to_json

from libs.gan_classifiers.GANomalyNetworks import ClassicalDenseNetworks, QuantumDecoderNetworks
from libs.gan_classifiers.Trainer import Trainer, QuantumDecoderTrainer
from libs.gan_classifiers.DataProcessor import Data
from libs.gan_classifiers.Plotter import Plotter, QuantumDecoderPlotter

gan_backends = {
    "classical": {"networks": ClassicalDenseNetworks,
                  "trainer": Trainer,
                  "plotter": Plotter, }
    ,
    "quantum": {"networks": QuantumDecoderNetworks,
                "trainer": QuantumDecoderTrainer,
                "plotter": QuantumDecoderPlotter, },
}

def run(data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Response:

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler("log.log", mode='w') 
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    """
    Default entry point of your code. Start coding here!

    Parameters:
        data (Optional[Dict[str, Any]]): The input data sent by the client
        params (Optional[Dict[str, Any]]): Contains parameters, which can be set by the client for
        parametrizing the execution

    Returns:
        response: (ResultResponse | ErrorResponse): Response as arbitrary json-serializable dict or an error to be
        passed back to the client
    """
    response: Response
    try:
        # Process data
        data = Data(pandas.DataFrame(data["values"], dtype="float64"), params)
        logger.info("Data loaded successfully.")

        # Load parameters and set defaults
        assert params["method"] in gan_backends.keys(), "No valid method parameter provided."
        assert params["train_or_predict"] in ["train", "predict"], "train_or_predict parameter not train or predict."
        logger.info("Parameters loaded successfully.")

        # Train or evaluate the classifier
        classifier = gan_backends[params["method"]]["networks"](data, params)
        trainer = gan_backends[params["method"]]["trainer"](data, classifier, params)
        #print("The following models will be used:")
        #classifier.print_model_summaries()

        if params["train_or_predict"] == "train":
            train_history = trainer.train()
            export_to_json(train_history["classifier"], "response_training.json")
            logger.info("Training of the GAN classifier has ended")        
            return ResultResponse(result=train_history["classifier"])
        elif params["train_or_predict"] == "predict":
            classifier.load(params)
            result = trainer.calculateMetrics(validation_or_test="test")
            export_to_json(result, "response_test.json")
            logger.info("Testing of the GAN classifier has ended")   
            return ResultResponse(result=result)
    except Exception as e:
        logger.error("An error occured while processing. Error reads:" '\n' + traceback.format_exc())
        return ErrorResponse(code="500", detail=f"{type(e).__name__}: {e}")

