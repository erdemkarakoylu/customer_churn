'''
This module contains functions required to run the Churn Prediction Project.
Functions include
* import_data
* perform_eda for Exploratory Data Analysis
* perform_feature_engineering
* encoder_helper (called by perform_feature_engineering)
* train_models
* classification_report_image (called by train_models)
* plot_roc (called by train_models)
* feature_importance_plot(called by train_models)
as well as model training and evaluation for the Churn Prediction Project.
'''
import os
import joblib
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import CAT_COLUMNS, KEEP_COLUMNS
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dframe: pandas dataframe
    '''
    dframe = pd.read_csv(pth)
    dframe['Churn'] = dframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return dframe


def perform_eda(dframe):
    '''
    perform eda on df and save figures to images folder
    input:
            dframe: pandas dataframe

    output:
            None
    '''
    fig = plt.figure(figsize=(20, 10))
    dframe['Churn'].hist()
    fig.savefig('./images/eda/churn_histogram.png')
    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    dframe['Customer_Age'].hist()
    fig.savefig('./images/eda/customer_age_hist.png')
    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    dframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    fig.savefig('./images/eda/marital_status_barplot.png')
    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.histplot(dframe['Total_Trans_Ct'], stat='density', kde=True)
    fig.savefig('./images/eda/total_trans_ct_density_plot.png')
    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(dframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig.savefig('./images/eda/correlation_heatmap.png')
    plt.close(fig)


def encoder_helper(dframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
            for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        category_lst = []
        encoded_name = category + '_' + response
        category_groups = dframe.groupby(category).mean()[response]
        for val in dframe[category]:
            category_lst.append(category_groups.loc[val])
        dframe[encoded_name] = category_lst
    return dframe


def perform_feature_engineering(dframe, response):
    '''
    input:
              dframe: pandas dataframe
              response: string of response name [optional argument that could be used
              for naming variables or index y column]

    output:
              X_train: pandas dataframe X training data
              X_test: pandas dataframe X testing data
              y_train: pandas dataframe y training data
              y_test: pandas dataframe y testing data
    '''
    df_encoded = encoder_helper(dframe, CAT_COLUMNS, response=response)
    inputs = pd.DataFrame()
    inputs[KEEP_COLUMNS] = df_encoded[KEEP_COLUMNS]
    output = dframe['Churn']
    x_train, x_test, y_train, y_test = train_test_split(
        inputs, output, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.text(0.01, 1.25, str('Random Forest Train'), {
            'fontsize': 10}, fontproperties='monospace')
    axis.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    axis.text(0.01, 0.6, str('Random Forest Test'), {
            'fontsize': 10}, fontproperties='monospace')
    axis.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    axis.set_visible('off')
    fig.savefig('./images/results/rf_cls_rep.png',)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5, 5))
    axis.text(0.01, 1.25, str('Logistic Regression Train'),
            {'fontsize': 10}, fontproperties='monospace')
    axis.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    axis.text(0.01, 0.6, str('Logistic Regression Test'), {
            'fontsize': 10}, fontproperties='monospace')
    axis.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    axis.set_visible('off')
    fig.savefig('./images/results/lr_cls_rep.png')
    plt.close(fig)


def plot_roc(x_data, y_data, model, out_pth):
    '''
    creates and stores the feature importances in out_pth
    input:
            x_data: pandas dataframe of X values
            y_data: pandas series of response value
            model: pre-trained model
            out_pth: path to store the figure

    output:
             None
    '''
    fig, axis = plt.subplots(figsize=(15, 8))
    plot_roc_curve(model, x_data, y_data, ax=axis)
    fig.savefig(out_pth, dpi=300)
    plt.close(fig)


def feature_importance_plot(rf_model, x_data, out_pth):
    '''
    creates and stores the feature importances in output_pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = rf_model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    fig, axis = plt.subplots(figsize=(20, 5))
    axis.set_title('Feature Importance')
    axis.set_ylabel('Importance')
    axis.bar(range(x_data.shape[1]), importances[indices])
    axis.set_xticks(range(x_data.shape[1]))
    axis.set_xticklabels(names, rotation=90)
    fig.savefig(out_pth)
    plt.close(fig)


def train_models(x_train, x_test, y_train, y_test, test_mode=False):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
              test_mode: False lets model training proceed
                         True loads previously trained models
    output:
              None
    '''
    if test_mode:
        rfc_model = joblib.load('./models/rfc_model.pkl')
        lr_model = joblib.load('./models/logistic_model.pkl')
    else:
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(x_train, y_train)
        # save best model
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        rfc_model = cv_rfc.best_estimator_

        lrc.fit(x_train, y_train)
        lr_model = lrc

    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)

    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)

    plot_roc(
        x_test, y_test, model=lr_model,
        out_pth="./images/results/lr_roc_curve.png")

    plot_roc(
        x_test, y_test, model=rfc_model,
        out_pth="./images/results/rf_roc_curve.png")

    feature_importance_plot(
        rfc_model, x_train, out_pth='./images/results/rf_feat_imp.png')

    classification_report_image(
        y_train=y_train, y_test=y_test, y_train_preds_lr=y_train_preds_lr,
        y_train_preds_rf=y_train_preds_rf, y_test_preds_lr=y_test_preds_lr,
        y_test_preds_rf=y_test_preds_rf)


if __name__ == "__main__":
    dataframe = import_data('./data/bank_data.csv')
    perform_eda(dataframe)
    dataframe = encoder_helper(dataframe, CAT_COLUMNS, response='Churn')
    X_train, X_test, Y_train, Y_test = perform_feature_engineering(
        dataframe, response='Churn')
    train_models(X_train, X_test, Y_train, Y_test)
