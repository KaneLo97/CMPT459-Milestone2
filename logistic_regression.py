from skimage.io import imread
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
import sys
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import statistics
import re
import scipy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression



def logistic_regression_classifier(train_df, test_df):
    X = train_df[['price', 'latitude', 'mean_des_tdidf', 
            'length_description', 'created_hour', 'closest_hospital',
            'closest_station', 'mean_feature_tdidf', 'created_day',
            'photos_num']]
    y = train_df['interest_level']
    
    X_test = test_df[['price', 'latitude', 'mean_des_tdidf', 
            'length_description', 'created_hour', 'closest_hospital',
            'closest_station', 'mean_feature_tdidf', 'created_day',
            'photos_num']]

    training_scores = []
    validation_scores = []
    training_logloss = []
    validation_logloss = []


    logreg = LogisticRegression(max_iter=10000)

    kf = KFold(n_splits=5, shuffle = False)
    X = np.array(X)
    y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_validation)
        y_pred1 = logreg.predict_proba(X_validation)
        training_scores.append(logreg.score(X_train, y_train))
        validation_scores.append(metrics.accuracy_score(y_validation, y_pred))
        validation_logloss.append(log_loss(y_validation, y_pred1))
    
    #check overfitting
    print('Scores from each Iteration: ', training_scores)
    print('Scores from each Iteration: ', validation_scores)
    print('Average k-fold on training: ', np.mean(training_scores))
    print('Average k-fold on validation: ', np.mean(validation_scores))
    print('Average k-fold on training using logloss: ', np.mean(validation_logloss))
    # print('Average k-fold on validation using logloss: ', np.mean(training_logloss))

    #train classifier
    logreg = logreg.fit(X,y)
    y_pred = logreg.predict_proba(X_test)
 
    submission = pd.DataFrame({
        "listing_id": test_df["listing_id"],
        "high": y_pred[:,0],
        "medium":y_pred[:,1],
        "low":y_pred[:,2]
    })

    titles_columns=["listing_id","high","medium","low"]
    submission=submission.reindex(columns=titles_columns)
    submission.to_csv('initial_submission_logistic.csv', index=False)

def improved_logistic_reg_classifier(train_df, test_df):
    print("RUN IMPORVED")
    X = train_df[['price', 'latitude', 'mean_des_tdidf', 
            'length_description', 'created_hour', 'closest_hospital',
            'closest_station', 'mean_feature_tdidf', 'created_day',
            'photos_num']]
    y = train_df['interest_level']
    
    X_test = test_df[['price', 'latitude', 'mean_des_tdidf', 
            'length_description', 'created_hour', 'closest_hospital',
            'closest_station', 'mean_feature_tdidf', 'created_day',
            'photos_num']]

    training_scores = []
    validation_scores = []
    training_logloss = []
    validation_logloss = []


    logreg = LogisticRegression(max_iter=100,C=0.001)
    
    kf = KFold(n_splits=5, shuffle = False)
    X = np.array(X)
    y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_validation)
        y_pred1 = logreg.predict_proba(X_validation)
        training_scores.append(logreg.score(X_train, y_train))
        # training_logloss.append(logreg.score(X_train, y_train))
        validation_scores.append(metrics.accuracy_score(y_validation, y_pred))
        validation_logloss.append(log_loss(y_validation, y_pred1))
    
    #check overfitting
    print('Scores from each Iteration: ', training_scores)
    print('Scores from each Iteration: ', validation_scores)
    print('Improved Average k-fold on training: ', np.mean(training_scores))
    print('Improved Average k-fold on validation: ', np.mean(validation_scores))
    print('Improved Average k-fold on training using logloss: ', np.mean(validation_logloss))
    # print('Average k-fold on validation using logloss: ', np.mean(training_logloss))

    #train classifier
    logreg = logreg.fit(X,y)
    y_pred = logreg.predict_proba(X_test)
 
    submission = pd.DataFrame({
        "listing_id": test_df["listing_id"],
        "high": y_pred[:,0],
        "medium":y_pred[:,1],
        "low":y_pred[:,2]
    })

    titles_columns=["listing_id","high","medium","low"]
    submission=submission.reindex(columns=titles_columns)
    submission.to_csv('initial_submission_logistic_improved.csv', index=False)

    # Code to hyperparameter tune
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # param_grid = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
    # clf = GridSearchCV(param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)
    # clf = GridSearchCV(LogisticRegression(max_iter=10000), param_grid=param_grid)
    # # Fit on data
    # clf.fit(X_train, y_train)
    # print("Best parameters set found on development set:")
    # print(clf.best_params_)
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))


def main():

    train_df = pd.read_json('new_train.json.zip')
    test_df = pd.read_json('new_test.json.zip')


    print(test_df)
    print(train_df)
    logistic_regression_classifier(train_df, test_df)
    improved_logistic_reg_classifier(train_df, test_df)
   


if __name__ == "__main__":
    main()
