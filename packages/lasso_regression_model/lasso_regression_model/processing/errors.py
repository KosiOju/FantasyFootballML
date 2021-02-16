import os
from lasso_regression_model.config import config as cfg


class BaseError(Exception):
	"""Base package error."""
	# this would be added to our "features.py" if we had one
	# how do we use this??

class InvalidModelInputError(BaseError):
	"""Model input contains an error."""


# not a class but this def is called in:
# test_pkl_file_persist.py
def pkl_file_list_count(*, test_dir):
	test_dir_file_list = []
	for file in os.listdir(test_dir):
			if file.endswith(".pkl"):
				test_dir_file_list.append(file)
	return len(test_dir_file_list)