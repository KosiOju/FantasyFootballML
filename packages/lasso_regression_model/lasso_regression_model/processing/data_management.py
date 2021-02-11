import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from lasso_regression_model.config import config as cfg


def load_dataset(*, file_name: str):
	_data = pd.read_csv(f"{cfg.DATASET_DIR}/{file_name}")
	
	return _data


def save_pipeline(*, pipeline_to_persist):
	# persist the model

	save_file_name = "lasso_regression.pkl"
	save_path = cfg.TRAINED_MODEL_DIR / save_file_name
	joblib.dump(pipeline_to_persist, save_path)

	print("saved pipeline")


def load_pipeline(*, file_name: str):
	# load persisted model

	file_path = cfg.TRAINED_MODEL_DIR / file_name
	saved_pipeline = joblib.load(filename=file_path)

	return saved_pipeline
