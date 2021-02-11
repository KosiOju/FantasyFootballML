import pandas as pd

from lasso_regression_model.processing.data_management import load_pipeline
from lasso_regression_model.config import config as cfg


pipeline_file_name = "lasso_regression.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data):
    # make prediction using the saved model pipeline

    data = pd.read_json(input_data)
    prediction = _price_pipe.predict(data[cfg.FEATURE_LIST])
    # if data was transformed the reverse would happen here
    response = {"predictions": prediction}

    return response
