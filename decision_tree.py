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

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import log_loss
import scipy
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_classif


def data_preprocessing(train_df):
    my_stop_words = list(stopwords.words('english'))

    #remove outliers
    train_df = train_df[train_df['bathrooms'] != 10.0]
    train_df = train_df[(train_df['latitude'] != 34.0126) | (train_df['latitude'] != 0)]
    train_df = train_df[(train_df['longitude'] > -80) & (train_df['longitude'] < -60)]
    train_df = train_df[train_df['price'] <1000000]

    train_df = train_df.apply(lambda row: missing_addr(row),axis=1)

    #remove records with missing street and display address
    train_df = train_df[(train_df['display_address'].apply(len)!= 0) & (train_df['street_address'].apply(len) != 0)]

    train_df = missing_features(train_df, my_stop_words)

    # Apply the function to remove tags
    train_df['description'] = train_df.apply(lambda row: remove_tags(row['description']),axis=1)

    # train_df = tf_idf(train_df, my_stop_words)

    return train_df
#
def missing_addr(row):

    if len(row.display_address) == 0:
        row.display_address = row.street_address
    return row

def missing_features(train_df, my_stop_words):
    descriptions = train_df
    features = train_df

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Fill in Missing Features using Description  #
    # # # # # # # # # # # # # # # # # # # # # # # #

    mask = features['features'].apply(len) == 0
    missing_features_df = features[mask]

    # Build a list of features in existing data
    df_has_features = features[~mask]
    all_lists = df_has_features['features'].to_numpy()
    feature_list = set()
    for each_list in all_lists:
        for elem in each_list:
            feature_list.add(elem)

    # Apply the function to fill missing features
    missing_features_df['features'] = missing_features_df.apply(lambda row: fill_features(row['description'],feature_list),axis=1)

    # Update the train_df with with empty 'features' columns
    train_df[train_df['features'].apply(len) == 0] = missing_features_df

    return train_df

# Parse the description column and find matching features
def fill_features(row, feature_list):
    result_list = []
    for feature in feature_list:
        if feature in row:
            result_list.append(feature)
    return result_list

def tf_idf(df, my_stop_words):
    # # # # # # # # # # # # # # # # #
    # Implement Feature Extraction  #
    # # # # # # # # # # # # # # # # #

    # Text Extraction for 'features' column
    features = df['features'].to_numpy()
    vectorizer_fea = CountVectorizer(stop_words=my_stop_words)
    corpus_fea = [" ".join(x) for x in features]
    X_fea_counts = vectorizer_fea.fit_transform(corpus_fea)

    # To TfIDF
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X_fea_counts)

    # Add the Mean Tf/Idf for each row in the database for Features
    num_of_rows = tfidf.shape[0]
    mean_val_list = []
    for row in range(num_of_rows):
        elem_list = tfidf[row].toarray()[0]
        row_vals = []
        mean_val = 0
        for elem in elem_list:
            if (float(elem) != float(0.0)):
                row_vals.append(float(elem))
        if len(row_vals) > 0:
            mean_val = statistics.mean(row_vals)
            mean_val_list.append(mean_val)
        else:
            mean_val_list.append(0.0)

    df['mean_feature_tdidf'] = mean_val_list

    # Text Extraction for 'description' column
    descriptions = df['description'].to_numpy()
    vectorizer_des = CountVectorizer(stop_words=my_stop_words)
    corpus_des = descriptions
    X_des_counts = vectorizer_des.fit_transform(corpus_des)
    # To TfIDF
    des_transformer = TfidfTransformer(smooth_idf=False)
    des_tfidf = des_transformer.fit_transform(X_des_counts)


    # # Add the Mean Tf/Idf for each row in the database for Description
    num_of_rows = des_tfidf.shape[0]
    mean_val_list = []
    for row in range(num_of_rows):
        elem_list = des_tfidf[row].toarray()[0]
        row_vals = []
        mean_val = 0
        for elem in elem_list:
            if (float(elem) != float(0.0)):
                row_vals.append(float(elem))
        if len(row_vals) > 0:
            mean_val = statistics.mean(row_vals)
            mean_val_list.append(mean_val)
        else:
            mean_val_list.append(0.0)

    df['mean_des_tdidf'] = mean_val_list

    # print(train_df)

    return df


# # # # # # # # # # # # # # # # #
# Remove HTML tags such as br   #
# # # # # # # # # # # # # # # # #
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

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


def decision_tree_classifier(train_df, test_df):
    feature_cols = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price', 'number_features', 'length_description', 'closest_station', 'closest_hospital', 'photos_num']
    x = train_df[feature_cols] # Features
    y = train_df.interest_level # Target variable
    X_test = test_df[feature_cols]

    # X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

    scores = []
    scores2 = []
    scores3 = []

    #create decision tree classifier
    # decision_tree = DecisionTreeClassifier(min_samples_split= 20, min_samples_leaf= 20)
    decision_tree = DecisionTreeClassifier()
    
    #cross validation -> choose the best fit classifier
    kf = KFold(n_splits=10, shuffle = False)
    X = np.array(x)
    y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        decision_tree = decision_tree.fit(X_train,y_train)
        y_pred = decision_tree.predict(X_validation)
        y_pred1 = decision_tree.predict_proba(X_validation)
        scores.append(decision_tree.score(X_validation, y_validation))
        scores2.append(metrics.accuracy_score(y_validation, y_pred))
        scores3.append((log_loss(y_validation, y_pred1), X_train, y_train, X_validation, y_validation))
    # print('Scores from each Iteration: ', scores)
    # print('Scores from each Iteration: ', scores2)
    # print('Scores from each Iteration: ', scores3)
    # print('Average k-fold score: ', np.mean(scores3))
    # print('Average k-fold score: ', np.mean(scores2))

    min_log_loss = min(scores3)
    X_train = min_log_loss[1]
    y_train = min_log_loss[2]
    X_validation = min_log_loss[3]
    y_validation = min_log_loss[4]

    print("Min log loss", min_log_loss[0])
    # print("x_train", X_train)
    # print("y_train", y_train)

    #train classifier
    decision_tree = decision_tree.fit(X_train,y_train)
    y_pred2 = decision_tree.predict_proba(X_train)
    y_pred3 = decision_tree.predict(X_train)

    print("Train Logloss:",log_loss(y_train, y_pred2))
    print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred3))

    y_pred1 = decision_tree.predict_proba(X_validation)
    y_pred2 = decision_tree.predict(X_validation)

    print("Test Logloss:",log_loss(y_validation, y_pred1))
    print("Test Accuracy:",metrics.accuracy_score(y_validation, y_pred2))

    y_pred = decision_tree.predict_proba(X_test)

    submission = pd.DataFrame({
        "listing_id": test_df["listing_id"],
        "high": y_pred[:,0],
        "medium":y_pred[:,1],
        "low":y_pred[:,2]
    })

    titles_columns=["listing_id","high","medium","low"]
    submission=submission.reindex(columns=titles_columns)
    submission.to_csv('initial_submission.csv', index=False)


def improved_decision_tree_classifier(train_df, test_df):
    # feature_cols = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price', 'number_features', 'length_description', 'closest_station', 'closest_hospital', 'created_month', 'created_day', 'created_hour', 'photos_num', 'mean_des_tdidf', 'mean_feature_tdidf']
    feature_cols = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price', 'number_features', 'length_description', 'created_day', 'created_hour', 'photos_num']
    x = train_df[feature_cols] # Features
    y = train_df.interest_level # Target variable
    X_test = test_df[feature_cols]

    # X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

    #max_depth = [3,5,3.5, 4, 4.5,5,5.5, 6, 6.5,7,8,9,10]
    # min_samples_split = [10,20,25,30,35,40]
    # min_samples_leaf = [10,20,25,30,35,40]

    # for depth in max_depth:
    #     decision_tree = DecisionTreeClassifier(criterion='gini',
    #                                max_depth = depth,
    #                                min_samples_split= 10,
    #                                min_samples_leaf= 5,
    #                                )
    #     results = cross_val_score(decision_tree, x, y, cv = 10, scoring = "accuracy")
    #     scores.append((results.mean(), depth))
    # print(scores)

    #create decision tree classifier
    decision_tree = DecisionTreeClassifier(criterion='gini',
                                   max_depth = 4.0,
                                   min_samples_split= 10,
                                   min_samples_leaf= 5,
                                   )
  
    scores = []
    scores2 = []
    scores3 = []

    kf = KFold(n_splits=10, shuffle = False)
    X = np.array(x)
    y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        decision_tree = decision_tree.fit(X_train,y_train)
        y_pred = decision_tree.predict(X_validation)
        y_pred1 = decision_tree.predict_proba(X_validation)
        scores.append(decision_tree.score(X_validation, y_validation))
        scores2.append(metrics.accuracy_score(y_validation, y_pred))
        scores3.append((log_loss(y_validation, y_pred1), X_train, y_train, X_validation, y_validation))
    # print('Scores from each Iteration: ', scores)
    # print('Scores from each Iteration: ', scores2)
    # print('Scores from each Iteration: ', scores3)
    # print('Average k-fold score: ', np.mean(scores3))
    # print('Average k-fold score: ', np.mean(scores2))

    min_log_loss = min(scores3)
    X_train = min_log_loss[1]
    y_train = min_log_loss[2]
    X_validation = min_log_loss[3]
    y_validation = min_log_loss[4]
    print("Min log loss", min_log_loss[0])
    # print("x train", X_train)
    # print("y train", y_train)
    

    # param_grid = dict(max_depth = max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    # random = RandomizedSearchCV(estimator=decision_tree, param_distributions=param_grid, cv = 3, n_jobs=-1)

    # random_result = random.fit(X_train,y_train)
    # # Summarize results
    # print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))

    #train classifier
    decision_tree = decision_tree.fit(X_train,y_train)

    # kfold = KFold(n_splits=5)
 
    # results_kfold = cross_val_score(decision_tree, x, y, cv=kfold)
    # print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
    # scores = cross_val_score(decision_tree, x, y, cv = 5, scoring = "neg_log_loss")
    # print(scores)
    # print('Average k-fold score: ', np.mean(scores))
    
    y_pred_train_prob = decision_tree.predict_proba(X_train)
    y_pred_train = decision_tree.predict(X_train)

    print("Train Improved Logloss:",log_loss(y_train, y_pred_train_prob))
    print("Train Improved Accuracy:",metrics.accuracy_score(y_train, y_pred_train))

    y_pred = decision_tree.predict_proba(X_validation)
    y_pred1 = decision_tree.predict(X_validation)
    

    print("Test Improved Logloss:",log_loss(y_validation, y_pred))
    print("Test Improved Accuracy:",metrics.accuracy_score(y_validation, y_pred1))

    y_pred = decision_tree.predict_proba(X_test)

    submission = pd.DataFrame({
        "listing_id": test_df["listing_id"],
        "high": y_pred[:,0],
        "medium":y_pred[:,1],
        "low":y_pred[:,2]
    })

    titles_columns=["listing_id","high","medium","low"]
    submission=submission.reindex(columns=titles_columns)
    submission.to_csv('improved_submission.csv', index=False)



def main():

    train_df = pd.read_json('train.json.zip')
    test_df = pd.read_json('test.json.zip')

    train_df = data_preprocessing(train_df)
    train_df = additionalFeatures(train_df)
    test_df = additionalFeatures(test_df)
   

    decision_tree_classifier(train_df, test_df)
    improved_decision_tree_classifier(train_df, test_df)




if __name__ == "__main__":
    main()