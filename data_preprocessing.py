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
    
    my_stop_words = list(stopwords.words('english'))
    df= tf_idf(df, my_stop_words)


    return df

def main():

    train_df = pd.read_json('train.json.zip')
    test_df = pd.read_json('test.json.zip')

    train_df = data_preprocessing(train_df)
    train_df = additionalFeatures(train_df)
    test_df = additionalFeatures(test_df)

    train_df.to_json("new_train.json.zip", compression='zip')
    test_df.to_json("new_test.json.zip", compression='zip')




if __name__ == "__main__":
    main()
