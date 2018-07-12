# !/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
created by Fangyang on Time:2018/7/11
'''

__author__ = 'Fangyang'

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y= digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_score = 0.0
best_k = -1
# for method in ['uniform', 'distance']:
for p in range(1, 6):   # p 只有在weights是distance时才有意义
    for k in range(1, 11):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights='distance', p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            # best_method = method
            best_p = p

# print("best_method : ", best_method)
print("best_k = ", best_k)
print("best_score = ", best_score)
print("best_p : ", best_p)

# '''
# best_k =  3
# best_score =  0.9888888888888889
# best_p :  3
# '''