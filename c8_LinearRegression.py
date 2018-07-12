# !/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
created by Fangyang on Time:2018/7/12
'''

__author__ = 'Fangyang'

import numpy as np
from c3_accuracy_score import r2_score


class LinearRegression(object):

    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def predict(self, x_predict):
        ''' 给定待测数据集X_predict, 返回表示X_predict的结果向量'''
        assert self.interception_ is not None and self.coef_ is not None,\
            'must fit before predict !'
        assert x_predict.shape[1] == len(self.coef_), \
            'the feature number of x_predict must be equal to x_train'

        X_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        ''' 根据测试数据集 X_test 和 y_test 确定当前模型的准确度'''
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"

    def fit_normal(self, X_train, y_train):
        ''' 根据训练数据集x_train, y_train训练 Linear Regression 模型'''
        assert X_train.shape[0] == y_train.shape[0], \
            'the size of the x_train must be equal to the size of y_train'

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv((X_b.T).dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self
