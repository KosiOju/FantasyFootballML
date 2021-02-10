import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin #REMEMBER
from sklearn.preprocessing import OrdinalEncoder


# THIS FILE DOESN'T INCLUDE TRAIN TEST SPLIT

# -------------------------------------------------------------------------
# __init__: ensures that strings are all converted into lists

# fit: what the class needs to learn from the data

# transform: transform a copy of the data and sub back into the real data
# -------------------------------------------------------------------------

# load features - can just use the config file



# feature engineer rare labels
class RareCategoricalVariables(BaseEstimator, TransformerMixin):
    # find frequent labels and replace rare ones with 'Rare'
    # self.variables __ CATEGORICAL_VARIABLES
    
    def __init__(self, variables=None, rare_perc=0.001):
        
        self.rare_perc = rare_perc
        
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        
        self.encoder_dict_ = {}
        
        for feature in self.variables:
            tmp = pd.Series(X[feature].value_counts(()) / float((len(X))))
            self.encoder_dict_[feature] = list(tmp[tmp >= self.rare_perc].index)
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(
                self.encoder_dict_[feature]), X[feature], 'Rare')
        print('Rare labels added...')
        
        return X



# ordinal encode cat vars
class OrdinalEncodeCategoricalVariables(BaseEstimator, TransformerMixin):
    # order and encode categorical variables
    # self.variables --> CATEGORICAL_VARIABLES
    
    def __init__(self, variables=None):
        
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y=None):
        # get_dummies isn't appropriate so use ordinal_map
        # add points column to X so groupby works!
        
        #X = X.copy()
        print()
        print(X.dtypes)
        
        self.enc = OrdinalEncoder()
        self.enc.fit(X[self.variables])
        
        return self
    
    def transform(self, X):
        
        X[self.variables] = self.enc.transform(X[self.variables])
        print()
        print(X.dtypes)
        
        return X
    
""" 
class DropTarget(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None): #feature_list = ???
            
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X):
        
        # X = X.copy()
        # we want to directly change and transform X
        
        X = X.drop(self.variables, axis=1)
        print('Target dropped from X...')
        
        return X
 """           
    
    
# Feature Scaling --> kinda the same as the train and transform scaler

# ... happens in the pipeline ...
# train and transform scaler

# fit and predict model