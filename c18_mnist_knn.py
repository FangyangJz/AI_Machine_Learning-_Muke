# !/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
created by Fangyang on Time:2018/7/19
'''

__author__ = 'Fangyang'

import numpy as np
import os
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
import time


path = os.getcwd()
mnist = fetch_mldata("mnist-original", data_home=path)
X, y = mnist['data'], mnist['target']

X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)

knn_clf = KNeighborsClassifier()
train_start_time = time.time()
knn_clf.fit(X_train, y_train)
train_stop_time = time.time()
print('train time is : ', (train_stop_time - train_start_time))

score_start_time = time.time()
result = knn_clf.score(X_test, y_test)
print('score is :', result)
score_stop_time = time.time()
print('score-process use time is : ', (score_stop_time - score_start_time))