from skimage.io import imread
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import log_loss
import scipy
# from sklearn.model_selection import cross_val_score
# from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV


def decision_tree_classifier(train_df, test_df):
    feature_cols = ['price', 'latitude', 'mean_des_tdidf', 
            'length_description', 'created_hour', 'closest_hospital',
            'closest_station', 'mean_feature_tdidf', 'created_day',
            'photos_num']
    x = train_df[feature_cols] # Features
    y = train_df.interest_level # Target variable
    X_test = test_df[feature_cols]

    
    # X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

    training_scores = []
    validation_scores = []
    training_logloss = []
    validation_logloss = []

    #create decision tree classifier
    # decision_tree = DecisionTreeClassifier(min_samples_split= 20, min_samples_leaf= 20)
    decision_tree = DecisionTreeClassifier()
    
    #cross validation 
    kf = KFold(n_splits=5, shuffle = False)
    X = np.array(x)
    y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        decision_tree = decision_tree.fit(X_train,y_train)
        y_pred = decision_tree.predict(X_validation)
        y_pred1 = decision_tree.predict_proba(X_validation)
        training_scores.append(decision_tree.score(X_train, y_train))
        training_logloss.append(decision_tree.score(X_train, y_train))
        validation_scores.append(metrics.accuracy_score(y_validation, y_pred))
        validation_logloss.append(log_loss(y_validation, y_pred1))
        # scores3.append((log_loss(y_validation, y_pred1), X_train, y_train, X_validation, y_validation))

    #check overfitting
    print('Scores from each Iteration: ', training_scores)
    print('Scores from each Iteration: ', validation_scores)
    print('Average k-fold on training: ', np.mean(training_scores))
    print('Average k-fold on testing: ', np.mean(validation_scores))
    print('Improved Average k-fold on training using logloss: ', np.mean(validation_logloss))
    print('Improved Average k-fold on validation using logloss: ', np.mean(training_logloss))


    # results = cross_val_score(decision_tree, x, y, cv = 5, scoring = "accuracy")
    # print (results)
    # print(np.mean(results))


    # min_log_loss = min(scores3)
    # X_train = min_log_loss[1]
    # y_train = min_log_loss[2]
    # X_validation = min_log_loss[3]
    # y_validation = min_log_loss[4]

    #print("Min log loss", min_log_loss[0])
    # print("x_train", X_train)
    # print("y_train", y_train)

    #train classifier
    decision_tree = decision_tree.fit(x,y)

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
    feature_cols = ['bedrooms','bathrooms','price', 'latitude', 'mean_des_tdidf', 'length_description', 'created_hour', 'closest_station', 'closest_hospital', 'mean_feature_tdidf', 'created_day','photos_num']
    x = train_df[feature_cols] # Features
    y = train_df.interest_level # Target variable
    X_test = test_df[feature_cols]

    # X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

    # Parameter Tuning
    # max_depth = [3,5,3.5, 4, 4.5,5,5.5, 6, 6.5,7,8,9,10]
    # min_samples_split = [10,20,25,30,35,40]
    # min_samples_leaf = [10,20,25,30,35,40]
    # parameters = [{'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}]
    # grid_search = GridSearchCV(estimator=decision_tree, param_grid=parameters, scoring ='accuracy',cv=5,n_jobs=-1)
    # grid_search = grid_search.fit(x,y)

    # best_accuracy = grid_search.best_score_
    # print(best_accuracy)
    # # best_accuracy
    # opt_param = grid_search.best_params_
    # print (opt_param )
     #create decision tree classifier
    decision_tree = DecisionTreeClassifier(criterion='gini',
                                   max_depth = 4.0,
                                   min_samples_split= 10,
                                   min_samples_leaf= 5,
                                   )
    
    training_scores = []
    validation_scores = []
    training_logloss = []
    validation_logloss = []

   #cross validation 
    kf = KFold(n_splits=5, shuffle = False)
    X = np.array(x)
    y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        decision_tree = decision_tree.fit(X_train,y_train)
        y_pred = decision_tree.predict(X_validation)
        y_pred1 = decision_tree.predict_proba(X_validation)
        training_scores.append(decision_tree.score(X_train, y_train))
        training_logloss.append(decision_tree.score(X_train, y_train))
        validation_scores.append(metrics.accuracy_score(y_validation, y_pred))
        validation_logloss.append(log_loss(y_validation, y_pred1))
    
    #check overfitting
    print('Scores from each Iteration: ', training_scores)
    print('Scores from each Iteration: ', validation_scores)
    print('Improved Average k-fold on training: ', np.mean(training_scores))
    print('Improved Average k-fold on validation: ', np.mean(validation_scores))
    print('Improved Average k-fold on training using logloss: ', np.mean(validation_logloss))
    print('Improved Average k-fold on validation using logloss: ', np.mean(training_logloss))



    #retrain classifier on the whole dataset
    decision_tree = decision_tree.fit(x,y)

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

def feature_Selection(train_df):
    cols = ['bedrooms', 'bathrooms', 'latitude', 
        'price', 'number_features', 
        'length_description', 'closest_station', 'closest_hospital',
        'created_month', 'created_day', 'created_hour', 'photos_num', 
        'mean_des_tdidf', 'mean_feature_tdidf']

    X = train_df[cols]
    y = train_df['interest_level']
    # Performing feature selection
    model = ExtraTreesClassifier()
    model.fit(X,y)
    #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    # plt.show()
    #print(feat_importances.nlargest(10))

def new_feature_Selection(train_df):
    cols = ['bedrooms', 'bathrooms', 'latitude', 
        'price', 'number_features', 
        'length_description', 'closest_station', 'closest_hospital',
        'created_month', 'created_day', 'created_hour', 'photos_num', 
        'mean_des_tdidf', 'mean_feature_tdidf']

    X = train_df[cols]
    y = train_df['interest_level']
    # Performing feature selection
    model = ExtraTreesClassifier()
    model.fit(X,y)
    #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(14).plot(kind='barh')
    # plt.show()
    # print(feat_importances.nlargest(14))

def main():

    train_df = pd.read_json('new_train.json.zip')
    test_df = pd.read_json('new_test.json.zip')

    # train_df = pd.read_json('train.json.zip')
    # test_df = pd.read_json('test.json.zip')


    # train_df = data_preprocessing(train_df)
    # train_df = additionalFeatures(train_df)
    # test_df = additionalFeatures(test_df)

    feature_Selection(train_df)
    decision_tree_classifier(train_df, test_df)

    new_feature_Selection(train_df)
    improved_decision_tree_classifier(train_df, test_df)




if __name__ == "__main__":
    main()