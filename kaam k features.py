# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:02:55 2018

@author: nagendra
"""

import pandas as pd
#import numpy as np
import sklearn.ensemble as ske
from sklearn import cross_validation, tree, linear_model
from sklearn.feature_selection import SelectFromModel



data = pd.read_csv('data.csv', sep='|')
x = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
y = data['legitimate'].values
#print (x)
#print (y)
fsel = ske.ExtraTreesClassifier().fit(x, y)

#print (fsel)

model = SelectFromModel(fsel, prefit=True)
#print (model)

x_new = model.transform(x)
#print (x_new)

a = x.shape
print (a)
b = x_new.shape
print (b)

