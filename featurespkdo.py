# -*- coding: utf-8 -*-
"""
Created on Sat March  20 23:31:40 2018

@author: ravindra
"""

import pandas as pd
import numpy as np
import sklearn.ensemble as ske
from sklearn import cross_validation, tree, linear_model
from sklearn.feature_selection import SelectFromModel


data = pd.read_csv('data.csv', sep='|')
x = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
y = data['legitimate'].values
print('Researching important feature based on %i total features\n' % x.shape[1])
fsel = ske.ExtraTreesClassifier().fit(x, y)
model = SelectFromModel(fsel, prefit=True)
x_new = model.transform(x)
a = x.shape
b = x_new.shape
print (a)
print (b)

nb_features = x_new.shape[1]
print (nb_features)
indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
print (indices)

for f in range(nb_features):
            print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], 
                  fsel.feature_importances_[indices[f]]))
