import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin #REMEMBER
from sklearn.preprocessing import OrdinalEncoder

from lasso_regression_model.processing.errors import InvalidModelInputError


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
    """String to numbers categorical encoder."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])["target"].mean().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # check if transformer introduces NaN
        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull().any()
            vars_ = {
                key: value for (key, value) in null_counts.items() if value is True
            }
            raise InvalidModelInputError(
                f"Categorical encoder has introduced NaN when "
                f"transforming categorical variables: {vars_.keys()}"
            )

        return X
    

"""
 I SHULD HAVE A GO AT ENCODING WITHOUT ORDINAL ENCODER
"""


# Feature Scaling --> kinda the same as the train and transform scaler

# ... happens in the pipeline ...
# train and transform scaler

# fit and predict model
