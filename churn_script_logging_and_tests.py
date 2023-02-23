from pathlib import Path
import logging

import numpy as np
import pytest

import churn_library as cl
from constants import CAT_COLUMNS, KEEP_COLUMNS

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
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
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.mark.parametrize(
    'eda_img_path',
    [
        'churn_histogram.png',
        'customer_age_hist.png',
        'marital_status_barplot.png',
        'total_trans_ct_density_plot.png',
        'correlation_heatmap.png'
    ]
)
def test_perform_eda(perform_eda, dataframe, image_dir, eda_img_path):
    '''
    test perform eda function.
    '''
    logging.info("Testing perform_eda...")
    perform_eda(dataframe)

    try:
        assert Path(image_dir + "/eda/" + eda_img_path).exists()
        logging.info(f"{eda_img_path} found.")
    except AssertionError as err:
        logging.error(f"{eda_img_path} does not appear to exist.")
        raise err


def test_encoder_helper(encoder_helper, dataframe, cat_cols):
    '''
    test encoder helper
    '''
    response = 'Churn'
    encoded_cols = [col + '_' + response for col in cat_cols]
    df = encoder_helper(dataframe, cat_cols, response=response)
    encoded_cols_set = set(encoded_cols)
    try:
        assert set(df.columns.to_list()).intersection(
            encoded_cols_set) == encoded_cols_set
        assert df[encoded_cols].shape[0] > 0
        logging.info("Testing encoder_helper: SUCCESS!")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Problem with the returned dataframe. ")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, dataframe):
    '''
    test perform_feature_engineering
    '''
    logging.info("Testing perform_feature_engineering...")
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        dataframe, response='Churn')

    try:
        assert X_train.size > 0
        assert X_test.size > 0
        assert y_train.size > 0
        assert y_test.size > 0
        logging.info("Success")
    except AssertionError as err:
        logging.error("Failure: returned data empty.")
        raise err


@pytest.mark.parametrize(
    'result_plot_path',
    [
        'lr_roc_curve.png',
        'rf_roc_curve.png',
        'lr_cls_rep.png',
        'rf_cls_rep.png',
        'rf_feat_imp.png'
    ]
)
def test_train_models(
    train_models, perform_feature_engineering, dataframe,
    logistic_model_path, random_forest_model_path,
    image_dir, result_plot_path
):
    '''
    test train_models.
    Note: adding "test mode" to train_models to avoid training models during testing.
    '''
    logging.info('Testing train_models...')
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        dataframe, response='Churn')
    train_models(X_train, X_test, y_train, y_test, test_mode=True)
    try:
        assert Path(logistic_model_path).exists()
        logging.info("Saved LR model file found.")
    except AssertionError as err:
        logging.error("LR model file does not appear to exist.")
        raise err
    try:
        assert Path(random_forest_model_path).exists()
        logging.info("Saved RF model file found.")
    except AssertionError as err:
        logging.error("RF model file does not appear to exist.")
        raise err
    try:
        assert Path(image_dir + '/results/' + result_plot_path).exists()
        logging.info(f'{result_plot_path} found.')
    except AssertionError as err:
        logging.error(f'{result_plot_path}does not appear to exist.')
        raise err


if __name__ == '__main__':
    pytest.main(['churn_script_logging_and_tests.py'])
