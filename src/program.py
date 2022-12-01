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
from libs.utilities import reformat_for_json, export_to_json

from libs.gan_classifiers.GANomalyNetworks import (
    ClassicalDenseClassifier,
    QuantumDecoderClassifier,
)

from libs.gan_classifiers.Metrics import UnsupervisedMetric, SupervisedMetric
from libs.gan_classifiers.Trainer import Trainer
from libs.gan_classifiers.DataProcessor import SupervisedData, UnsupervisedData

gan_backends = {
    "classical": ClassicalDenseClassifier,
    "quantum": QuantumDecoderClassifier,
}


def run(
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Response:

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler("log.log", mode="w")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
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
    try:
        # Process data & load metrics
        if params["is_supervised"]:
            data_values = SupervisedData(
                pandas.DataFrame(data["values"], dtype="float64"), params
            )
            metrics_object = SupervisedMetric(data_values, params)
        else:
            data_values = UnsupervisedData(
                pandas.DataFrame(data["values"], dtype="float64"), params
            )
            metrics_object = UnsupervisedMetric(data_values, params)
        logger.info("Data loaded successfully.")

        # Load parameters and set defaults
        assert (
            params["method"] in gan_backends.keys()
        ), "No valid method parameter provided."
        assert params["train_or_predict"] in [
            "train",
            "predict",
        ], "train_or_predict parameter not train or predict."
        if params["train_or_predict"] == "predict":
            assert (
                "trained_model" in params
            ), "The model was run in predict mode but there is no model information under trained_model in the parameters."
        logger.info("Parameters loaded successfully.")

        # generate the network
        classifier = gan_backends[params["method"]](data_values, params)

        if params["train_or_predict"] == "train":
            trainer = Trainer(data_values, classifier, metrics_object, params)
            classifier_weights = trainer.train()
            output = metrics_object.get_last_metrics()
            output["trained_model"] = classifier_weights
            output = reformat_for_json(output)
            export_to_json(output, "response_training.json")
            logger.info("Training of the GAN classifier has ended")
            return ResultResponse(result=output)
        elif params["train_or_predict"] == "predict":
            classifier.load(params["trained_model"])
            result = metrics_object.calculate_metrics(
                data_values.get_test_data(),
                classifier.predict,
                classifier.generate,
            )
            output = reformat_for_json(result)
            export_to_json(output, "response_test.json")
            logger.info("Testing of the GAN classifier has ended")
            return ResultResponse(result=output)
    except Exception as e:
        logger.error(
            "An error occured while processing. Error reads:"
            "\n" + traceback.format_exc()
        )
        return ErrorResponse(code="500", detail=f"{type(e).__name__}: {e}")
