# Customer Churn Prediction

## Project Description

The goal of this project is the development of a churn predictor for credit card users. To that end, a [kaggle data set](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code) was used. Running the code will have the following effects:

-   load the data
-   run an EDA on the data and produce relevant plots.
-   engineer features for predictive modeling
-   train two models; a logistic regression classifier, and a random forest classifier.
-   produce evalution plots for the trained models.

## Files and data description

The project directory is structured as follows. Scripts are found in the project root directory. **churn_library.py** is the main script file for data loading, EDA, feature engineering, model training, and model evaluation. **churn_script_logging_and_tests.py** contains tests for functions in **churn_library.py**; these tests were written for pytest.
Data is located in the data/ directory stored as CSV.

Subfolders were named after their content. See below for details

```
.  
├── README.md
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── constants.py
├── data
│  └── bank_data.csv
├── images
│   ├── eda
│   │   ├── churn_histogram.png
│   │   ├── correlation_heatmap.png
│   │   ├── customer_age_hist.png
│   │   ├── marital_status_barplot.png
│   │   └── total_trans_ct_density_plot.png
│   └── results
│   ├── lr_cls_rep.png
│   ├── lr_roc_curve.png
│   ├── rf_cls_rep.png
│   ├── rf_feat_imp.png
│   └── rf_roc_curve.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── pytest.ini
├── requirements_py3.6.txt
├── requirements_py3.8.txt
└── requirements_py3.8.yml
```
9 directories, 28 files

## Requirements
Requirements come in 3 versions. requirements_py36.txt and py38.txt are for installation via pip, while requirements_py38.yml are geared toward conda.

Libraries required:
    - jobling\
    - matplotlib\
    - numpy\
    - pandas\
    - scikit-learn\
    - seaborn\
    - shap\
    - pytest
    
## Running Files

**churn_library.py** can be run with <code>python churn_library.py</code>. No command line options are available.

**churn_script_logging_and_testing.py** can be run either with
<code>pytest churn_script_logging_and_testing.py</code> or with
<code>ipython churn_script_logging_and_testing.py</code>. Note the **pytest.ini** file in the project directory, which contains a flag to turn off *pytest* logging. This is to allow the logger, instantiated in **churn_script_logging_and_testing.py**, to take effect and store logs in the ./logs directory.