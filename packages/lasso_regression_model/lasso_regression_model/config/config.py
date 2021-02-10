import pathlib

from sklearn.linear_model import Lasso

PACKAGE_ROOT = pathlib.Path(lasso_regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# data
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
TARGET = 'points'

# # PIPELINE_NAME = 'lasso_regression.pkl'
# # FEATURE_LIST_FILE = 'selected_features.csv'


RARE_PERC = 0.001

CATEGORICAL_VARIABLES = ['playerName']

NUMERICAL_VARIABLES = [] # not needed here

FEATURE_LIST = ['minsPlayed', 'goalsScored', 'assists', 'cleanSheets',
                'goalsConceded', 'ownGoals', 'penSaved', 'yelCards', 'redCards',
                'saves', 'bonus', 'influence', 'creativity', 'threat',
                'costGBP', 'playerName']
    # will have to manually add this
    # points and oppositionTeam needs dropping as well

# # do we need a num_na_not_allowed and cat_na_not_allowed
                
            