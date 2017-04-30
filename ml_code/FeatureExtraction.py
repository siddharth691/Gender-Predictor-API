#Importing modules

import pandas as pd
import numpy as np
import random
from nltk import NaiveBayesClassifier,classify
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier

class FeatureExtraction:
"""
Feature extraction module
"""
  def __init__(self, dataframe):
      self.dataframe = dataframe

  def countVowels(self, string):
      num_vowels=0
      for char in string:
          if char in "aeiou":
             num_vowels = num_vowels+1
      return num_vowels

  def countEI(self, string):
      num_ei=0
      for char in string:
          if char in "ei":
             num_ei = num_ei+1
      return num_ei

  def _nameFeatures(self, name):
      name=name.lower()
      return{'name':name,
          'lastChar':name[-1],
            'lastTwoChar':name[-2:],
            'isLastAEIY':(name[-1] in 'aeiy'),
            'isSecLastAEI':(name[-2] in 'aei'),
            'NoOfVowels':self.countVowels(name),
             'NumEI':self.countEI(name),
             'length':len(name),
             'firstChar':name[0]
             }
  def extractFeatures(self):
      self.featureSet = list()
      for index,row in self.dataframe.iterrows():
          self.featureSet.append((self._nameFeatures(row['name']),row['gender']))
      return self.featureSet

  def TrainAndTestNB(self):
      self.featureSet = self.extractFeatures(self.dataframe)
      random.shuffle(self.featureSet)
      name_count = len(self.featureSet)
      cut = int(name_count*0.80)
      self.trainSet = self.featureSet[:cut]
      self.testSet = self.featureSet[cut:]
      self.classifier = NaiveBayesClassifier.train(self.trainSet)
      print('Testing Accuracy: {} '.format(classify.accuracy(self.classifier,self.testSet)))
      print('Most Informative Features')
      print(self.classifier.show_most_informative_features(5))

  def TrainAndTestSVM(self):
      self.featureSet = self.extractFeatures(self.dataframe)
      random.shuffle(self.featureSet)
      name_count = len(self.featureSet)
      cut = int(name_count*0.80)
      self.trainSet = self.featureSet[:cut]
      self.testSet = self.featureSet[cut:]
      classif= SklearnClassifier(LinearSVC())
      self.classifier = classif.train(self.trainSet)
      print('Testing Accuracy: {} '.format(classify.accuracy(self.classifier,self.testSet)))
