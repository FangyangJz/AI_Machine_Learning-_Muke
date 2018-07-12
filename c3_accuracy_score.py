# !/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
created by Fangyang on Time:2018/7/11
'''
from math import sqrt
import numpy as np

__author__ = 'Fangyang'

def accuracy_score(y_true, y_predict):

    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict)/len(y_true)

def mean_squared_error(y_true, y_predict):
    '''计算y_true和y_predict之间的MSE'''
    assert len(y_true) == len(y_predict), \
        'the size of y_true must be equal to the size of y_predict'
    return np.sum((y_predict - y_true)**2)/len(y_true)

def root_mean_squared_error(y_true, y_predict):
    '''计算y_true和y_predict之间的RMSE'''
    return sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    '''计算y_true和y_predict之间的MAE'''
    assert len(y_true) == len(y_predict), \
        'the size of y_true must be equal to the size of y_predict'
    return np.sum(np.absolute(y_predict - y_true)) /len(y_true)

def r2_score(y_true, y_predict):
    '''计算 R square'''
    return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)