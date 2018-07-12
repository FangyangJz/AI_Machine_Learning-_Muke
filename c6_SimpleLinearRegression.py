# !/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
created by Fangyang on Time:2018/7/12
'''

__author__ = 'Fangyang'

import numpy as np
from c3_accuracy_score import r2_score


class SimpleLinearRegression1:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):

        assert x_train.ndim == 1, \
            'Simple linear regression can only solve single feature training data'
        assert len(x_train) == len(y_train), \
            'the size of x_train must be equal to the size of y_train'

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0  # 分子
        deno = 0.0  # 分母
        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            deno += (x_i - x_mean) ** 2

        self.a_ = num/deno
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):

        assert x_predict.ndim == 1, \
            'Simple linear regressor can only solve single feature training data'
        assert self.a_ is not None and self.b_ is not None,\
            'must fit before predict'

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_* x_single + self.b_

    def __repr__(self):
        return "Simple Linear Regression1()"


class SimpleLinearRegression2:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):

        assert x_train.ndim == 1, \
            'Simple linear regression can only solve single feature training data'
        assert len(x_train) == len(y_train), \
            'the size of x_train must be equal to the size of y_train'

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0  # 分子
        deno = 0.0  # 分母
        # for x_i, y_i in zip(x_train, y_train):
        #     num += (x_i - x_mean) * (y_i - y_mean)
        #     deno += (x_i - x_mean) ** 2
        num = (x_train - x_mean).dot(y_train - y_mean)
        deno = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num/deno
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):

        assert x_predict.ndim == 1, \
            'Simple linear regressor can only solve single feature training data'
        assert self.a_ is not None and self.b_ is not None,\
            'must fit before predict'

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_* x_single + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "Simple Linear Regression2()"