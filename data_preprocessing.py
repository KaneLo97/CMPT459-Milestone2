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
    
    train_df['number_features'] = train_df['features'].apply(len)
    train_df['length_description'] = train_df['description'].apply(len)

    train_df = missing_features(train_df, my_stop_words)
    
    # Apply the function to remove tags
    train_df['description'] = train_df.apply(lambda row: remove_tags(row['description']),axis=1)

    #tf_idf(train_df, my_stop_words)
    
    return train_df

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

def tf_idf(train_df, my_stop_words):
    # # # # # # # # # # # # # # # # # 
    # Implement Feature Extraction  # 
    # # # # # # # # # # # # # # # # # 

    # Text Extraction for 'features' column
    features = train_df['features'].to_numpy()
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

    train_df['mean_feature_tdidf'] = mean_val_list

    # Text Extraction for 'description' column
    descriptions = train_df['description'].to_numpy()
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

    train_df['mean_des_tdidf'] = mean_val_list
    print(train_df)

    
    
# # # # # # # # # # # # # # # # # 
# Remove HTML tags such as br   # 
# # # # # # # # # # # # # # # # # 
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

def main():

    train_df = pd.read_json('train.json.zip')
    test_df = pd.read_json('test.json.zip')
    print (train_df)
    train_df = data_preprocessing(train_df)
    print (train_df)


if __name__ == "__main__":
    main()