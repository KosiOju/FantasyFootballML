
from lasso_regression_model.config import config as cfg
from lasso_regression_model.processing.data_management import load_dataset



def test_xtrain_column_length():
	# given
	test_data = load_dataset(file_name='xtrain.csv')
	test_data_column_len = len(test_data.columns) - 6

	# when
	correct_column_len = len(cfg.FEATURE_LIST)

	# then
	assert test_data_column_len is not None
	assert test_data_column_len == correct_column_len

def test_xtest_column_length():
	# given
	test_data = load_dataset(file_name='xtest.csv')
	test_data_column_len = len(test_data.columns) - 6

	# when
	correct_column_len = len(cfg.FEATURE_LIST)

	# then
	assert test_data_column_len is not None
	assert test_data_column_len == correct_column_len

def test_xtrain_larger_than_xtest():
	# given
	test_xtrain_data = load_dataset(file_name='xtrain.csv')
	test_xtrain_data_size = len(test_xtrain_data.index)

	# when
	test_xtest_data = load_dataset(file_name='xtest.csv')
	test_xtest_data_size = len(test_xtest_data.index)

	# then
	assert test_xtrain_data_size is not None
	assert test_xtest_data_size is not None

	assert test_xtrain_data_size > test_xtest_data_size