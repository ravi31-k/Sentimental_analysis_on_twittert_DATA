# -*- coding: utf-8 -*-
"""
Created on Sat April  28 10:50:57 2018

@author: ravindra
"""
import pandas as pd
import numpy as np
import sklearn.ensemble as ske
from sklearn import cross_validation, tree, linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('data.csv', sep='|')
x = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
y = data['legitimate'].values

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_new, y ,test_size=0.2)

fsel = ske.ExtraTreesClassifier().fit(x, y)
model = SelectFromModel(fsel, prefit=True)
x_new = model.transform(x)
a = x.shape
b = x_new.shape
nb_features = x_new.shape[1]
indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]


algorithms = {
        "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
        "RandomForest": ske.RandomForestClassifier(n_estimators=50),
        "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50),
        "AdaBoost": ske.AdaBoostClassifier(n_estimators=100),
        "GNB": GaussianNB()
    }

results = {}
print("\nNow testing algorithms")
for algo in algorithms:
    clf = algorithms[algo]
    #print (clf)
    y= clf.fit(x_train, y_train)
    #print (y)
    score = clf.score(x_test, y_test)
   # print (score)
    print("%s : %f %%" % (algo, score*100))
    results[algo] = score
    
winner = max(results, key=results.get)
print('\nWinner algorithm is %s with a %f %% success' % (winner, results[winner]*100))


