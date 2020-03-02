from skimage.io import imread
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

def logistic_regression_classifier(train_df, test_df):
    X_train = train_df['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price', 'number_features', 'length_description', 'closest_station', 'closest_hospital', 'created_month', 'created_day', 'created_hour', 'photos_num', 'mean_des_tdidf', 'mean_feature_tdidf']
    y_train = train_df['interest_level']
    
    X_test = test_df['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price', 'number_features', 'length_description', 'closest_station', 'closest_hospital', 'created_month', 'created_day', 'created_hour', 'photos_num', 'mean_des_tdidf', 'mean_feature_tdidf']

    scores = []
    scores2 = []
    scores3 = []

    #cross validation -> choose the best fit classifier
    kf = KFold(n_splits=10, shuffle = False)
    X = np.array(x)
    y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_validation)
        y_pred1 = logreg.predict_proba(X_validation)
        scores.append(logreg.score(X_validation, y_validation))
        scores2.append(metrics.accuracy_score(y_validation, y_pred))
        scores3.append((log_loss(y_validation, y_pred1), X_train, y_train, X_validation, y_validation))


    print('Scores from each Iteration: ', scores)
    print('Scores from each Iteration: ', scores2)
    print('Scores from each Iteration: ', scores3)
    print('Average k-fold score: ', np.mean(scores3))
    print('Average k-fold score: ', np.mean(scores2))

def main():
    train_df = pd.read_json('new_train.json.zip')
    test_df = pd.read_json('new_test.json.zip')

    test_df.head()
    logistic_regression_classifier(train_df, test_df)

if __name__ == "__main__":
    main()