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

# Import from Kane's
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression


def additionalFeatures(df):
    station = pd.read_csv("nyc-transit-data.csv")
    station = station.filter(['Station Name', 'Station Latitude', 'Station Longitude'])
    station.drop_duplicates(inplace=True)

    # Clean up code for hospital data
    point_of_intrest = pd.read_csv("Point_Of_Interest.csv")
    point_of_intrest =  point_of_intrest[point_of_intrest['NAME'].str.contains("HOSPITAL")]
    point_of_intrest['the_geom'] = point_of_intrest['the_geom'].str.split(' ')
    point_of_intrest['latitude'] = point_of_intrest['the_geom'].apply(lambda x: x[1])
    point_of_intrest['longitude'] = point_of_intrest['the_geom'].apply(lambda x: x[0])
    # print(point_of_intrest['latitude'])
    # print(point_of_intrest['longitude'])

    # Get distance to hospitals
    mat_hospital = scipy.spatial.distance.cdist(df[['latitude','longitude']],
                                point_of_intrest[['latitude','longitude']], metric='euclidean')

    min_distance_to_hospital = []
    for listing in mat_hospital:
        min_distance_to_hospital.append(min(listing))

    df['closest_hospital'] = min_distance_to_hospital

    # Get distance to subway stations
    mat = scipy.spatial.distance.cdist(df[['latitude','longitude']],
                                station[['Station Latitude','Station Longitude']], metric='euclidean')

    min_distance_to_station = []

    for listing in mat:
        min_distance_to_station.append(min(listing))

    df['closest_station'] = min_distance_to_station

    df['number_features'] = df['features'].apply(len)
    df['length_description'] = df['description'].apply(len)
    df['created'] = pd.to_datetime(df['created'])
    df['created_hour'] = df["created"].dt.hour
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df['photos_num'] = df['photos'].apply(len)


    return df


def logistic_regression_classifier(train_df, test_df):
    X = train_df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price', 'number_features', 'length_description', 'closest_station', 'closest_hospital', 'photos_num']]
    y = train_df['interest_level']
    
    X_test = test_df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price', 'number_features', 'length_description', 'closest_station', 'closest_hospital', 'photos_num']]

    # y_test = 

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # logreg = LogisticRegression(max_iter=10000)
    # logreg.fit(X_train, y_train)

    # # Use score method to get accuracy of model
    # score = logreg.score(X_test, y_test)
    # print(score)

    ####################
    scores = []
    scores2 = []
    scores3 = []


    #cross validation -> choose the best fit classifier
    kf = KFold(n_splits=10, shuffle = False)
    X = np.array(X)
    y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_validation)
        y_pred1 = logreg.predict_proba(X_validation)
        scores.append(logreg.score(X_validation, y_validation))
        scores2.append(metrics.accuracy_score(y_validation, y_pred))
        scores3.append((log_loss(y_validation, y_pred1), X_train, y_train, X_validation, y_validation))

    min_log_loss = min(scores3)
    X_train = min_log_loss[1]
    y_train = min_log_loss[2]
    X_validation = min_log_loss[3]
    y_validation = min_log_loss[4]

    print("Min log loss", min_log_loss[0])
    # print("x_train", X_train)
    # print("y_train", y_train)

    #train classifier
    logreg = logreg.fit(X_train,y_train)
    y_pred2 = logreg.predict_proba(X_train)
    y_pred3 = logreg.predict(X_train)

    print("Train Logloss:",log_loss(y_train, y_pred2))
    print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred3))

    y_pred1 = logreg.predict_proba(X_validation)
    y_pred2 = logreg.predict(X_validation)

    print("Test Logloss:",log_loss(y_validation, y_pred1))
    print("Test Accuracy:",metrics.accuracy_score(y_validation, y_pred2))

    y_pred = logreg.predict_proba(X_test)

    submission = pd.DataFrame({
        "listing_id": test_df["listing_id"],
        "high": y_pred[:,0],
        "medium":y_pred[:,1],
        "low":y_pred[:,2]
    })

    titles_columns=["listing_id","high","medium","low"]
    submission=submission.reindex(columns=titles_columns)
    submission.to_csv('initial_submission.csv', index=False)

def improved_logistic_reg_classifier(train_df, test_df):
    print("RUN IMPORVED")
    X = train_df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price', 'number_features', 'length_description', 'closest_station', 'closest_hospital', 'photos_num']]
    y = train_df['interest_level']

        ####################
    scores = []
    scores2 = []
    scores3 = []


    #cross validation -> choose the best fit classifier
    kf = KFold(n_splits=10, shuffle = False)
    X = np.array(X)
    y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        logreg = LogisticRegression(max_iter=1000,penalty='l2',C=100)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_validation)
        y_pred1 = logreg.predict_proba(X_validation)
        scores.append(logreg.score(X_validation, y_validation))
        scores2.append(metrics.accuracy_score(y_validation, y_pred))
        scores3.append((log_loss(y_validation, y_pred1), X_train, y_train, X_validation, y_validation))

    min_log_loss = min(scores3)
    X_train = min_log_loss[1]
    y_train = min_log_loss[2]
    X_validation = min_log_loss[3]
    y_validation = min_log_loss[4]

    print("Min log loss", min_log_loss[0])
    # print("x_train", X_train)
    # print("y_train", y_train)

    #train classifier
    logreg = logreg.fit(X_train,y_train)
    y_pred2 = logreg.predict_proba(X_train)
    y_pred3 = logreg.predict(X_train)

    print("Train Logloss:",log_loss(y_train, y_pred2))
    print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred3))

    y_pred1 = logreg.predict_proba(X_validation)
    y_pred2 = logreg.predict(X_validation)

    print("Test Logloss:",log_loss(y_validation, y_pred1))
    print("Test Accuracy:",metrics.accuracy_score(y_validation, y_pred2))


    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
    # clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
    # # Fit on data
    # clf.fit(X_train, y_train)
    # print("Best parameters set found on development set:")
    # print(clf.best_params_)
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))


def main():

    train_df = pd.read_json('new_train.json.zip')
    test_df = pd.read_json('test.json.zip')

    test_df = additionalFeatures(test_df)

    print(test_df)
    print(train_df)
    logistic_regression_classifier(train_df, test_df)
    improved_logistic_reg_classifier(train_df, test_df)
   






if __name__ == "__main__":
    main()
