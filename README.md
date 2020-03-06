# CMPT459-Milestone2


## Purpose

### data_preprocessing.py
The code in this Jupyter Notebook implements preprocessing of the data. In addition, new attrributes are calculated and addted to both training and testing dataset.
Running the code produces two JSON files in the zip format. These files are further used to perform train classifiers. The "new_test.json.zip" file is not included in this
repository as it exceeds maximum file size of Github (100 MB). 

1) new_train.json.zip
2) new_test.json.zip

### decision_tree.py
Trains and tests the Decision Tree classifier. 

### logistic_regression.py
Trains and tests the Logistic Regression classifier. 

### svm.ipynb
Trains and tests the SVM classifier. 

## Order of operation

The order of operation is relevant.
The code in "data_preprocessing.py" should be run first so that the classifiers are able to use generated files to train and test the models..

## Running Python Files


Running data_preprocessing.py
`python3 data_preprocessing.py`

Running decision_tree.py
`python3 decision_tree.py`

Running logistic_regression.py
`python3 logistic_regression.py`

Running svm.ipynb
1. Start Jupyter Notebook server
`$ jupyter notebook`
2. Click on `svm.ipynb` to open the file
3. Run the file by `Kernal->Restart & Run All`



## Libraries Used
- sklearn
- skimage
- numpy
- pandas
- seaborn
- scipy
- matplotlib
- sklearn
- nltk

