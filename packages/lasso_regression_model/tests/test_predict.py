import math

from lasso_regression_model.predict import make_prediction
from lasso_regression_model.processing.data_management import load_dataset


def test_make_single_prediction():
	# given
	test_data = load_dataset(file_name='xtest.csv')
	single_test_json = test_data[0:1].to_json(orient='records')
	# reads the 1st line of 'test.csv'

	# when
	subject = make_prediction(input_data=single_test_json)
	# print(subject.get('predictions')[0])
	# the above print statement was used to get the correct predicted value for
	# line 22 ~ subject to change after editing

	# then
	assert subject is not None
	assert isinstance(subject.get('predictions')[0], float)
	assert math.ceil(subject.get('predictions')[0]) == 16 # see line 17
	