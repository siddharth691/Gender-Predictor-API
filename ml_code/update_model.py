import pandas as pd
import numpy as np
import random
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.externals import joblib
from ml_code import FeatureExtraction
import os
import logging


def update_model(ml_code_loc):
    #Create logging for update model function
    logger = logging.getLogger('update_model')
    # Load cleaned data
    cleanedData = pd.read_csv('/home/ubuntu/Gender-Predictor-API/ml_code/CleanedData.csv', header =0)
    #Load update data
    update_data = pd.read_csv('/home/ubuntu/Gender-Predictor-API/ml_code/update_data.csv', header =None, names=['name', 'gender'])
    
    #concatenate cleaned data and update data

    update_data = pd.concat([cleanedData, update_data], axis=0)

    fx = FeatureExtraction.FeatureExtraction(update_data)
    featureSet = fx.extractFeatures()
    random.shuffle(featureSet)


    clf = SklearnClassifier(LinearSVC())  
    clf.train(featureSet)

    _model_DIR = '/home/ubuntu/Gender-Predictor-API/model/model.pkl'
    #model_filename = os.path.join(_model_DIR, "model.pkl")

    joblib.dump(clf, _model_DIR)
    logger.info("Successfully updated the model and Pickled into models folder %s",_model_DIR)
