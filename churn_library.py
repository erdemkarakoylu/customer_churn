# library doc string


# import libraries
import joblib
import os
os.environ['QT_QPA_PLATFORM']='offscreen'


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

from constants import CAT_COLUMNS, KEEP_COLUMNS


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    f = plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    f.savefig('./images/eda/churn_histogram.png')
    plt.close(f)
    f = plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    f.savefig('./images/eda/customer_age_hist.png')
    plt.close(f)
    f = plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    f.savefig('./images/eda/marital_status_barplot.png')
    plt.close(f)
    f=plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    f.savefig('./images/eda/total_trans_ct_density_plot.png')
    plt.close(f)
    f=plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    f.savefig('./images/eda/correlation_heatmap.png')
    plt.close(f)


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        category_lst = []
        encoded_name = category + '_' + response
        category_groups = df.groupby(category).mean()[response]
        for val in df[category]:
            category_lst.append(category_groups.loc[val])
        df[encoded_name] = category_lst
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: pandas dataframe X training data
              X_test: pandas dataframe X testing data
              y_train: pandas dataframe y training data
              y_test: pandas dataframe y testing data
    '''
    df_encoded = encoder_helper(df, CAT_COLUMNS, response='Churn')
    X = pd.DataFrame()
    X[KEEP_COLUMNS] = df_encoded[KEEP_COLUMNS]
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


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
    f, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    ax.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    ax.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    ax.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    ax.set_visible('off')
    f.savefig('./images/results/rf_cls_rep.png',)
    plt.close(f)

    f, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    ax.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    ax.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    ax.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    ax.set_visible('off')
    f.savefig('./images/results/lr_cls_rep.png')
    plt.close(f)


def plot_roc(X_data, y_data, model, out_pth):
    '''
    creates and stores the feature importances in out_pth
    input:
            X_data: pandas dataframe of X values
            y_data: pandas series of response value
            model: pre-trained model
            out_pth: path to store the figure

    output:
             None
    '''
    f, ax = plt.subplots(figsize=(15, 8))
    plot_roc_curve(model, X_data, y_data, ax=ax)
    f.savefig(out_pth, dpi=300)
    plt.close(f)


def feature_importance_plot(rf_model, X_data, out_pth):
    '''
    creates and stores the feature importances in output_pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances=rf_model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    f, ax = plt.subplots(figsize=(20, 5))
    ax.set_title('Feature Importance')
    ax.set_ylabel('Importance')
    ax.bar(range(X_data.shape[1]), importances[indices])
    ax.set_xticks(range(X_data.shape[1]))
    ax.set_xticklabels(names, rotation=90)
    f.savefig(out_pth)
    plt.close(f)


def train_models(X_train, X_test, y_train, y_test, test_mode=False):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
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
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
        }
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)
        # save best model
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        rfc_model = cv_rfc.best_estimator_

        lrc.fit(X_train, y_train)
        lr_model = lrc
    
    y_train_preds_rf = rfc_model.predict(X_train)
    y_test_preds_rf = rfc_model.predict(X_test)

    y_train_preds_lr = lr_model.predict(X_train)
    y_test_preds_lr = lr_model.predict(X_test)

    plot_roc(
        X_test, y_test, model=lr_model, 
        out_pth="./images/results/lr_roc_curve.png")
    
    plot_roc(
        X_test, y_test, model=rfc_model, 
        out_pth="./images/results/rf_roc_curve.png")

    feature_importance_plot(
        rfc_model, X_train, out_pth='./images/results/rf_feat_imp.png')
    
    classification_report_image(
        y_train=y_train, y_test=y_test, y_train_preds_lr=y_train_preds_lr, 
        y_train_preds_rf=y_train_preds_rf, y_test_preds_lr=y_test_preds_lr, 
        y_test_preds_rf=y_test_preds_rf)


if __name__ == "__main__":
    df = import_data('./data/bank_data.csv')
    perform_eda(df)
    df = encoder_helper(df, CAT_COLUMNS, response='Churn')
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, response='Churn')
    train_models(X_train, X_test, y_train, y_test)
        
        
