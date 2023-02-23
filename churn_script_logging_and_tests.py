"""This module contains tests for all functions in churn_library.py.
Tests were written for pytest and can be invoked from the command line either with
pytest or (i)python."""
from pathlib import Path
import logging

import pytest


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(funcName)s - %(message)s')


def test_import(import_data, df_path):
    '''
    Test of import_data function in churn_library.py.
    '''
    try:
        dframe = import_data(df_path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dframe.shape[0] > 0
        assert dframe.shape[1] > 0
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
    Test of perform_eda function.
    '''
    logging.info("Testing perform_eda...")
    perform_eda(dataframe)

    try:
        assert Path(image_dir + "/eda/" + eda_img_path).exists()
        logging.info(" %s found.", eda_img_path)
    except AssertionError as err:
        logging.error("%s does not appear to exist.", eda_img_path)
        raise err


def test_encoder_helper(encoder_helper, dataframe, cat_cols):
    '''
    Test of encoder_helper function in churn_library.py.
    '''
    response = 'Churn'
    encoded_cols = [col + '_' + response for col in cat_cols]
    dframe = encoder_helper(dataframe, cat_cols, response=response)
    encoded_cols_set = set(encoded_cols)
    try:
        assert set(dframe.columns.to_list()).intersection(
            encoded_cols_set) == encoded_cols_set
        assert dframe[encoded_cols].shape[0] > 0
        logging.info("Testing encoder_helper: SUCCESS!")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Problem with the returned dataframe. ")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, dataframe):
    '''
    Test of perform_feature_engineering function in churn_library.py.
    '''
    logging.info("Testing perform_feature_engineering...")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        dataframe, response='Churn')

    try:
        assert x_train.size > 0
        assert x_test.size > 0
        assert y_train.size > 0
        assert y_test.size > 0
        logging.info("Success")
    except AssertionError as err:
        logging.error("Failure: returned data empty.")
        raise err



def test_model_files(logistic_model_path, random_forest_model_path):
    "Testing for the availability of saved model files."
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
    image_dir, result_plot_path
):
    '''
    Test of train_models function in churn_library.py.
    Note: adding "test mode" to train_models to avoid training models during testing.
    '''
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        dataframe, response='Churn')
    train_models(x_train, x_test, y_train, y_test, test_mode=True)

    try:
        assert Path(image_dir + '/results/' + result_plot_path).exists()
        logging.info('%s found.', result_plot_path)
    except AssertionError as err:
        logging.error('%s does not appear to exist.', result_plot_path)
        raise err


if __name__ == '__main__':
    pytest.main(['churn_script_logging_and_tests.py'])
