import pandas as pd

import joblib
import config as cfg

def make_prediction(input_data):
    
    _titanic_pipe = joblib.load(filename= cfg.PIPELINE_NAME)
    
    results = _titanic_pipe.predict(input_data)
    
    return results



if __name__ == '__main__':
    
    # test pipeline
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import r2_score
    
    data = pd.read_csv(cfg.TRAINING_DATA_FILE)
    
    X_train, X_test, y_train, y_test = train_test_split(
        data[cfg.FEATURE_LIST],
        data[cfg.TARGET],
        test_size=0.2,
        random_state=0)
    
    pred = make_prediction(X_test)
    
    # the less the mse the more efficient the model is
    print('Mean Squared Error: {}'.format(mse(y_test, pred)))
    print('Mean Error: {}'.format(mse(y_test, pred, squared=False)))
    # correlation (1) close to 1 or -1 are good! - 0 is bad!
    # r2 is easier to understand than regular r
    # r2 shows how much of the data variation is explained by the fitted line
    print('r2 score: {}'.format(r2_score(y_test, pred)))
    print()
    