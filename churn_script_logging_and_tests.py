from pathlib import Path
import logging

import numpy as np
import pytest

import churn_library as cl
from constants import CAT_COLUMNS, KEEP_COLUMNS

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(funcName)s - %(message)s')


@pytest.fixture(scope='session')
def lr_roc_curve_path():
	return './images/results/lr_roc_curve.png'

@pytest.fixture(scope='session')
def rf_roc_curve_path():
	return './images/results/rf_roc_curve.png'

@pytest.fixture(scope='session')
def lr_cls_rep_path():
	return './images/results/lr_cls_rep.png'

@pytest.fixture(scope='session')
def rf_cls_rep_path():
	return './images/results/rf_cls_rep.png'

@pytest.fixture(scope='session')
def rf_feat_imp_path():
	return './images/results/rf_feat_imp.png'

@pytest.fixture(scope='session')
def logistic_model_path():
	return './models/logistic_model.pkl'

@pytest.fixture(scope='session')
def random_forest_model_path():
	return './models/rfc_model.pkl'

@pytest.fixture(scope='session')
def keep_cols():
	return KEEP_COLUMNS

@pytest.fixture(scope='session')
def cat_cols():
	return CAT_COLUMNS

@pytest.fixture(scope="session")
def image_dir():
	return "./images"

@pytest.fixture(scope="session")
def df_path():
	return "./data/bank_data.csv"

@pytest.fixture(scope="session")
def dataframe(df_path):
	return cl.import_data(df_path)

@pytest.fixture
def train_models():
	return cl.train_models

@pytest.fixture
def perform_feature_engineering():
	return cl.perform_feature_engineering

@pytest.fixture
def encoder_helper():
	return cl.encoder_helper

@pytest.fixture
def perform_eda():
	return cl.perform_eda

@pytest.fixture
def import_data():
	return cl.import_data


def test_import(import_data, df_path):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data(df_path)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda, dataframe, image_dir):
	'''
	test perform eda function.
	'''
	perform_eda(dataframe)
	assert Path(image_dir + "/eda/churn_histogram.png").exists()
	assert Path(image_dir + "/eda/customer_age_hist.png").exists()
	assert Path(image_dir + "/eda/marital_status_barplot.png").exists()
	assert Path(image_dir + "/eda/total_trans_ct_density_plot.png").exists()
	assert Path(image_dir + "/eda/correlation_heatmap.png").exists()

	

def test_encoder_helper(encoder_helper, dataframe, cat_cols):
	'''
	test encoder helper
	'''
	response = 'Churn'
	encoded_cols = [col + '_' + response for col in cat_cols]
	df = encoder_helper(dataframe, cat_cols, response=response)
	encoded_cols_set = set(encoded_cols)
	assert set(df.columns.to_list()).intersection(encoded_cols_set) == encoded_cols_set
	assert df[encoded_cols].shape[0] > 0



def test_perform_feature_engineering(perform_feature_engineering, dataframe):
	'''
	test perform_feature_engineering
	'''
	X_train, X_test, y_train, y_test = perform_feature_engineering(
		dataframe, response='Churn')
	assert X_train.size > 0 
	assert X_test.size > 0
	assert y_train.size > 0
	assert y_test.size > 0
	assert set(X_train.columns.to_list()) == set(KEEP_COLUMNS)
	assert set(X_test.columns.to_list()) == set(KEEP_COLUMNS)
	assert y_train.name == 'Churn'
	assert y_test.name == 'Churn'



def test_train_models(
		train_models, perform_feature_engineering, dataframe, 
		logistic_model_path, random_forest_model_path,
		lr_roc_curve_path, rf_roc_curve_path, lr_cls_rep_path, 
		rf_cls_rep_path, rf_feat_imp_path
		):
	'''
	test train_models.
	Note: adding "test mode" to train_models to avoid training models during testing.
	'''
	X_train, X_test, y_train, y_test = perform_feature_engineering(
		dataframe, response='Churn')
	train_models(X_train, X_test, y_train, y_test, test_mode=True)
	try:
		assert Path(logistic_model_path).exists()
		logging.info("LR model binary found.")
	except AssertionError as err:
		logging.error("LR model binary file does not appear to exist.")
		raise err
	try:
		assert Path(random_forest_model_path).exists()
		logging.info("RF model binary file found.")
	except AssertionError as err:
		logging.error("RF model binary file does not appear to exist.") 
		raise err
	try:
		assert Path(lr_roc_curve_path).exists()
		logging.info("LR ROC curve plot found.")
	except AssertionError as err:
		logging.error("LR ROC curve plot does not appear to exist.")
		raise err
	try:
		assert Path(rf_roc_curve_path).exists()
		logging.info("RF ROC curve plot found.")
	except AssertionError as err:
		logging.error("RF ROC curve does not appear to exist.")
		raise err
	try:
		assert Path(rf_feat_imp_path).exists()
		logging.info("RF feature importance plot found.")
	except AssertionError as err:
		logging.error("RF feature importance plot does not appear to exist.")
		raise err
	try:
		assert Path(lr_cls_rep_path).exists()
		logging.info('LR classification report image found.')
	except AssertionError as err:
		logging.error('LR classification report image does not appear to exist.')
		raise err
	try:
		assert Path(rf_cls_rep_path).exists()
		logging.info('RF classification report image found.')
	except AssertionError as err:
		logging.error('RF classification report image does not appear to exist.')
		raise err

if __name__ == "__main__":
	pass








