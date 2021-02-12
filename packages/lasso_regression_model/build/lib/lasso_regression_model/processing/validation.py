from lasso_regression_model.config import config as cfg

import pandas as pd


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
	# -> shows what type the returned variable will be
	# check model inputs for unprocessable values

	validated_data = input_data.copy()

	# check for NA in num vars not spotted during training
	if input_data[cfg.NUMERICAL_VARIABLES].isnull().any().any():
		validated_data = validated_data.dropna(
			axis=0, subset=cfg.NUMERICAL_VARIABLES
			)

	# check for NA in cat vars
	if input_data[cfg.CATEGORICAL_VARIABLES].isnull().any().any():
		validated_data = validated_data.dropna(
			axis=0, subset=cfg.CATEGORICAL_VARIABLES
			)


	# dont need to check for 0<= values

	return validated_data