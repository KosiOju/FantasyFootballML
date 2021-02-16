import numpy as np

from lasso_regression_model.processing.data_management import load_dataset


# need to test how the encode_dict in preprocessors.py
# to check it is not empty
# not sure how to do this yet...


def test_xtest_playerNname_dtype():
	# given
	test_data_xtest = load_dataset(file_name='xtest.csv')

	# when
	correct_dtype = np.dtype(np.int64)
	test_dtype_xtest = test_data_xtest["playerName"].dtypes

	# then
	assert test_dtype_xtest == correct_dtype

def test_xtrain_playerName_dtype():
	# given
	test_data_xtrain = load_dataset(file_name='xtrain.csv')

	# when
	correct_dtype = np.dtype(np.int64)
	test_dtype_xtrain = test_data_xtrain["playerName"].dtypes

	# then
	assert test_dtype_xtrain == correct_dtype