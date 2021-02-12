import pathlib

import lasso_regression_model

import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(lasso_regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# data
TESTING_DATA_FILE = "xtest.csv"
TRAINING_DATA_FILE = "xtrain.csv"
TARGET = 'points'

# # PIPELINE_NAME = 'lasso_regression.pkl'
# # FEATURE_LIST_FILE = 'selected_features.csv'


RARE_PERC = 0.001

CATEGORICAL_VARIABLES = ['playerName']

FEATURE_LIST = ['minsPlayed', 'goalsScored', 'assists', 'cleanSheets',
                'goalsConceded', 'ownGoals', 'penSaved', 'yelCards', 'redCards',
                'saves', 'bonus', 'influence', 'creativity', 'threat',
                'costGBP', 'playerName']
    # will have to manually add this
    # points and oppositionTeam needs dropping as well

NUMERICAL_VARIABLES = [
	var for var in FEATURE_LIST
	if var not in CATEGORICAL_VARIABLES	
	]

# # do we need a num_na_not_allowed and cat_na_not_allowed
# # not needed for me...


PIPELINE_NAME = 'lasso_regression' # ERROR POSIBILITY
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.5