# !/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
created by Fangyang on Time:2018/7/11
'''
from collections import Counter
from math import sqrt
import numpy as np

__author__ = 'Fangyang'

class KNNClassifier:

    def __init__(self, k):
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):

        assert X_train.shape[0] == y_train.shape[0], \
            'the size of X_tain must be equal to y_train'
        assert self.k <= X_train.shape[0], \
            'the size of X_train must be at least k'

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):

        assert self._X_train is not None and self._y_train is not None, \
            "must fit befor predict"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):

        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k = %d)"%self.k

    def score(self, X_test, y_test):

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)  # 这个函数在c3_accuracy_score里面写的