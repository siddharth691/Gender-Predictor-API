#Importing modules and dataset
import pandas as pd
import numpy as np
import random
from nltk import NaiveBayesClassifier,classify
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
dfM = pd.read_csv('male',header=0)
dfF = pd.read_csv('female',header=0)

#Completely removing samples that donot provide first name
dfM.drop([2016,3623,12070,6385,7797,13494,10909,14196],axis=0, inplace=True)
dfF.drop([4419,1406,3570,10347,2861,14336],axis=0,inplace=True)
#Combining male and female dataset and shuffling the rows
combined = pd.concat([dfM, dfF], axis = 0)
combined = combined.iloc[(np.random.permutation(len(combined)))]
combined.reset_index(drop = True, inplace = True)

#Wrangling with the dataset

##Dropping the unnecessary extra column
combined.drop('race',axis = 1, inplace =True)
# Drop NaN (missing names)
combined = combined.dropna(axis=0)

##Removing duplicate rows
##Dataset almost reduces to half when duplicate rows are removed
# combined = combined.drop_duplicates(subset = ['name','gender'])
combined.reset_index(drop = True, inplace = True)

## Removing @ and keeping first part
combined.name = combined.name.map(lambda name: name.split('@')[0].strip() if(name.find('@')!=-1) else name.strip())


##Removing titles ( they appear before dot)
combined.name = combined.name.map(lambda name: name.strip() if(name.find('.')==-1) else name.split('.')[1].strip() if(len(name.split('.')[0].strip())<=5) else name.strip())

##Removing titles (if they don't have dot)
def checkTitle1(name):
    title = ['j ','dr ','ku ','ku- ','k ','km ','km- ','kum ','km0 ','sant ','st ','mo ','gen ','smt ','ms ','mis ','shri ','sri ','sh ','shi ','p ','md ','gd ','m ','sk ','so ','mohd ','mho ','dd ','ed ','ct ','na ', 'miss ', 'lc ', 'smt- ', 'smts ','smt-', 'smt,','1-smt ','mo- ','gs-1957975 ','mrs ','shrimati ','a ','b ']
    k = any(name.find(i)==0 for i in title)
    return k
def checkTitle2(name):
    title = ['ku  ','kum  ','shri  ','md  ','mohd  ','smt  ','km  ']
    k = any(name.find(i)==0 for i in title)
    return k
def checkTitle3(name):
    title = ['s p ','kum a ','ct b ']
    k = any(name.find(i)==0 for i in title)
    return k
combined.name = combined.name.map(lambda name: name.split('  ')[1].strip() if(checkTitle2(name)==True) else name.split(' ')[2].strip() if (checkTitle3(name)==True) else name.split(' ')[1].strip() if(checkTitle1(name)==True) else name.strip())


##Extracting first name 
combined.name = combined.name.map(lambda name: name if(name.find(' ')==-1) else name.split()[0].strip())


##Removing any special character associated with the first name
combined.name = combined.name.map(lambda name: ''.join(i for i in name if i.isalpha()).strip())

## Again Removing NAN rows
combined = combined.dropna(axis=0)

##Removing non english rows
for index,row in combined.iterrows():
    try:
        row['name'].encode('ascii')
    except UnicodeEncodeError:
        combined.drop(index,axis=0,inplace=True)
combined.reset_index(drop = True, inplace = True)

##Again removing duplicate rows
# combined = combined.drop_duplicates(subset = ['name','gender'])
combined.reset_index(drop = True, inplace = True)

## copying data to csv file
combined.to_csv('CleanedData.csv', index=False)