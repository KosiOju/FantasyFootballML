# ============= PATHS =========
# load the data to save the pipeline

TRAINING_DATA_FILE = 'ffmlDf_20-21'
PIPELINE_NAME = 'lasso_regression.pkl'
FEATURE_LIST_FILE = 'selected_features.csv'

# ======= FEATURE GROUPS =============
# list of vars needed for the different engineering steps
# or groups of variables that we want to engineer

TARGET = 'points'

RARE_PERC = 0.001

CATEGORICAL_VARIABLES = ['playerName'] # , 'oppositionTeam']

NUMERICAL_VARIABLES = [] # not needed here

FEATURE_LIST = ['minsPlayed', 'goalsScored', 'assists', 'cleanSheets',
                'goalsConceded', 'ownGoals', 'penSaved', 'yelCards', 'redCards',
                'saves', 'bonus', 'influence', 'creativity', 'threat',
                'costGBP', 'playerName']
    # will have to manually add this
    # points and oppositionTeam needs dropping as well
                
            