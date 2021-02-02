from sklearn.linear_model import Lasso # model
from sklearn.pipeline import Pipeline # sklearn.pipeline
from sklearn.preprocessing import MinMaxScaler # scaler

import preprocessors as pps
import config as cfg


ffml_pipe = Pipeline(
    # complete with the list of steps from the pps file
    # and the list of vars from cfg
    [
     ('ReplaceRareCategoricalVariables',
      pps.RareCategoricalVariables(
          variables = cfg.CATEGORICAL_VARIABLES,
          rare_perc = cfg.RARE_PERC)),
     
     ('OrdinalEncodeCategoricalVariables',
      pps.OrdinalEncodeCategoricalVariables(
          variables = cfg.CATEGORICAL_VARIABLES)),
     
     
     # need:
         # Transform - num vars dont get transformed in this one
         # Drop features - dropped in pp.CategoricalEncoder
         # Scaler
         # model fitting
           
      ('Scaler',
       MinMaxScaler()),
      
      ('LinearModel',
       Lasso(alpha=0.005, random_state=0))
      
     ]
    )