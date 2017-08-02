import pandas as pd
import numpy as np
import random
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.externals import joblib
import FeatureExtraction
import os
import logging


def update_model(ml_code_loc)
    #Create logging for update model function
    logger = logging.getLogger('update_model')
    logger.setLevel(logging.INFO)

    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename = 'update_model.log', format=FORMAT)
    # Load cleaned data
    cleanedData = pd.read_csv('CleanedData.csv', header =0)
    #Load update data
    update_data = pd.read_csv(ml_code_loc+'update_data.csv', header =None, names=['name', 'gender'])
    
    #concatenate cleaned data and update data

    update_data = pd.concat([cleanedData, update_data], axis=0)

    fx = FeatureExtraction.FeatureExtraction(update_data)
    featureSet = fx.extractFeatures()
    random.shuffle(featureSet)


    clf = SklearnClassifier(LinearSVC())  
    clf.train(featureSet)

    _model_DIR = os.path.dirname(os.path.realpath('model.pkl'))

    model_filename = os.path.join(_model_DIR, "model.pkl")

    joblib.dump(clf, model_filename)
    logger.info("Successfully updated the model and Pickled into models folder")