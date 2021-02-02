import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from pipeline import ffml_pipe
import config as cfg



def run_training():
    print('Training the model...')
    
    # read training data
    data = pd.read_csv(cfg.TRAINING_DATA_FILE)
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[cfg.FEATURE_LIST],
        data[cfg.TARGET],
        test_size=0.2,
        random_state=0)
    
    ffml_pipe.fit(X_train, y_train)
    joblib.dump(ffml_pipe, cfg.PIPELINE_NAME)
    
    print('Model trained...')
    
    

if __name__ == '__main__':
    run_training()