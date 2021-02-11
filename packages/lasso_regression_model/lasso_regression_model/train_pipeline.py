import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from lasso_regression_model import pipeline
from lasso_regression_model.config import config as cfg


def save_pipeline(*, pipeline_to_persist):
    # persist the pipeline

    save_file_name = "lasso_regression.pkl"
    save_path = cfg.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

    print("saved pipeline")



def run_training():
    print('Training the model...')
    
    # read training data
    data = pd.read_csv(cfg.DATASET_DIR / cfg.TRAINING_DATA_FILE)
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[cfg.FEATURE_LIST],
        data[cfg.TARGET],
        test_size=0.2,
        random_state=0) # setting the seed here
    
    # if data tranformed then the target would be transformed
    # here as well

    pipeline.ffml_pipe.fit(X_train[cfg.FEATURE_LIST], y_train)
    save_pipeline(pipeline_to_persist=pipeline.ffml_pipe)
    
    print('Model trained...')
    
    

if __name__ == '__main__':
    run_training()