from sklearn.model_selection import train_test_split

from lasso_regression_model import pipeline
from lasso_regression_model.processing.data_management import load_dataset, save_pipeline
from lasso_regression_model.config import config as cfg


def run_training():
    print('Training the model...')
    
    # read training data
    data = load_dataset(file_name=cfg.TRAINING_DATA_FILE)
    
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