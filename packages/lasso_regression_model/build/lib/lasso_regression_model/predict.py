import pandas as pd

from lasso_regression_model.processing.data_management import load_pipeline
from lasso_regression_model.config import config as cfg
from lasso_regression_model.processing.validation import validate_inputs
from lasso_regression_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)

pipeline_file_name = f"{cfg.PIPELINE_SAVE_FILE}{_version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data):
    # make prediction using the saved model pipeline

    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)
    prediction = _price_pipe.predict(data[cfg.FEATURE_LIST])
    # if data was transformed the reverse would happen here
    
    results = {"predictions": prediction, "version": _version}

    _logger.info(
    	f"Making predictions with model version: {_version}"
    	f"Inputs: {validated_data}"
    	f"Predictions: {results}"
    )

    return results
