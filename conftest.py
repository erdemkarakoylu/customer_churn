"""
This file contains pytest fixtures for churn_script_logging_and_tests.py.

Author: Erdem Karaköylü
Creation Date: 2/14/2023
"""

import pytest
import churn_library as cl
from constants import CAT_COLUMNS, KEEP_COLUMNS


@pytest.fixture(scope='session')
def lr_roc_curve_path():
    """Pytest fixture for Logistic Regression ROC plot path."""
    return './images/results/lr_roc_curve.png'


@pytest.fixture(scope='session')
def rf_roc_curve_path():
    """Pytest fixture for Random Forest ROC plot path."""
    return './images/results/rf_roc_curve.png'


@pytest.fixture(scope='session')
def lr_cls_rep_path():
    "Pytest fixture for LR classification report image path."
    return './images/results/lr_cls_rep.png'


@pytest.fixture(scope='session')
def rf_cls_rep_path():
    "Pytest fixture for RF classification report image path."
    return './images/results/rf_cls_rep.png'


@pytest.fixture(scope='session')
def rf_feat_imp_path():
    "Pytest fixture for RF feature importance plot path."
    return './images/results/rf_feat_imp.png'


@pytest.fixture(scope='session')
def logistic_model_path():
    "Pytest fixture for LR model file path."
    return './models/logistic_model.pkl'


@pytest.fixture(scope='session')
def random_forest_model_path():
    "Pytest fixture for RF model file path."
    return './models/rfc_model.pkl'


@pytest.fixture(scope='session')
def keep_cols():
    "Pytest fixture for input column to keep for modeling."
    return KEEP_COLUMNS


@pytest.fixture(scope='session')
def cat_cols():
    "Pytest fixture for categorical columns in input data."
    return CAT_COLUMNS


@pytest.fixture(scope="session")
def image_dir():
    "Pytest fixture for path to images directory."
    return "./images"


@pytest.fixture(scope="session")
def df_path():
    "Pytest fixture for path to data used for modeling."
    return "./data/bank_data.csv"


@pytest.fixture(scope="session")
def dataframe(df_path):
    "Pytest fixture for dataframe containing data."
    return cl.import_data(df_path)


@pytest.fixture
def train_models():
    "Pytest fixture for train_models function in churn_library.py."
    return cl.train_models


@pytest.fixture
def perform_feature_engineering():
    "Pytest fixture for perform_feature_engineering func. in churn_library.py."
    return cl.perform_feature_engineering


@pytest.fixture
def encoder_helper():
    "Pytest fixture for encoder_helper function in churn_library.py."
    return cl.encoder_helper


@pytest.fixture
def perform_eda():
    "Pytest fixture for perform_eda function in churn_library.py"
    return cl.perform_eda


@pytest.fixture
def import_data():
    "Pytest fixture for import_data function in churn_library.py"
    return cl.import_data