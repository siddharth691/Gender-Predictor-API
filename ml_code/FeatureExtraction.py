def countVowels(string):
    num_vowels=0
    for char in string:
        if char in "aeiou":
           num_vowels = num_vowels+1
    return num_vowels

def countEI(string):
    num_ei=0
    for char in string:
        if char in "ei":
           num_ei = num_ei+1
    return num_ei

def _nameFeatures(name):
    name=name.lower()
    return{'name':name,
        'lastChar':name[-1],
          'lastTwoChar':name[-2:],
          'isLastAEIY':(name[-1] in 'aeiy'),
          'isSecLastAEI':(name[-2] in 'aei'),
          'NoOfVowels':countVowels(name),
           'NumEI':countEI(name),
           'length':len(name),
           'firstChar':name[0]
           }
def extractFeatures(dataframe):
    featureSet = list()
    for index,row in dataframe.iterrows():
        featureSet.append((_nameFeatures(row['name']),row['gender']))
    return featureSet

def TrainAndTestNB(dataframe):
    featureSet = extractFeatures(dataframe)
    random.shuffle(featureSet)
    name_count = len(featureSet)
    cut = int(name_count*0.80)
    trainSet = featureSet[:cut]
    testSet = featureSet[cut:]
    classifier = NaiveBayesClassifier.train(trainSet)
    print('Testing Accuracy: {} '.format(classify.accuracy(classifier,testSet)))
    print('Most Informative Features')
    print(classifier.show_most_informative_features(5))

def TrainAndTestSVM(dataframe):
    featureSet = extractFeatures(dataframe)
    random.shuffle(featureSet)
    name_count = len(featureSet)
    cut = int(name_count*0.80)
    trainSet = featureSet[:cut]
    testSet = featureSet[cut:]
    classif = SklearnClassifier(LinearSVC())
    classifier = classif.train(trainSet)
    print('Testing Accuracy: {} '.format(classify.accuracy(classifier,testSet)))
