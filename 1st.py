# -*- coding: utf-8 -*-
"""
Created on Sat March 14  17:42:10 2018

@author: ravindra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv', sep='|')
legit_binaries = data[0:41323].drop(['legitimate'], axis=1)
malicious_binaries = data[41323::].drop(['legitimate'], axis=1)

print (legit_binaries)
print (malicious_binaries)

x = legit_binaries['FileAlignment'].value_counts()
y = malicious_binaries['FileAlignment'].value_counts()
'''
print (x)
print (y)
'''
c = plt.hist([legit_binaries['SectionsMaxEntropy'], malicious_binaries['SectionsMaxEntropy']], 
             range=[0,8], normed=True, color=["green", "red"],label=["legitimate", "malicious"])
print (c)
d = plt.legend()
print (d)
e = plt.show()
print (e)
