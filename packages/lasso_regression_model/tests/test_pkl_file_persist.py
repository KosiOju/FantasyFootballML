from lasso_regression_model.config import config as cfg
from lasso_regression_model.processing import errors


def test_only_one_pkl_file_saved_in_trained_models():
	# given
	test_dir = cfg.TRAINED_MODEL_DIR

	# when
	test_dir_file_list_count = errors.pkl_file_list_count(test_dir=test_dir)


	# then
	assert test_dir_file_list_count is not None
	assert test_dir_file_list_count == 1