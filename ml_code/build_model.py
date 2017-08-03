import pandas as pd
import numpy as np
import random
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.externals import joblib
import FeatureExtraction
import os
import logging


if __name__ == "__main__":
    #Create logging for build model function
    logger = logging.getLogger('build_model')
    logger.setLevel(logging.INFO)

    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename = '/home/ubuntu/Gender-Predictor-API/ml_code/build_model.log', format=FORMAT)

    
    # Load cleaned data
    cleanedData = pd.read_csv('/home/ubuntu/Gender-Predictor-API/ml_code/CleanedData.csv', header =0)
    fx = FeatureExtraction.FeatureExtraction(cleanedData)
    featureSet = fx.extractFeatures()
    random.shuffle(featureSet)


    clf = SklearnClassifier(LinearSVC())  
    clf.train(featureSet)

    _CUR_DIR = '/home/ubuntu/Gender-Predictor-API/ml_code'
    _SERIALIZATION_DIR = os.path.join(_CUR_DIR, "..", "model")

    if not os.path.exists(_SERIALIZATION_DIR):
        os.makedirs(_SERIALIZATION_DIR)
    model_filename = os.path.join(_SERIALIZATION_DIR, "model.pkl")

    joblib.dump(clf, model_filename)
    logger.info("Successfully Built and Pickled into models folder")
