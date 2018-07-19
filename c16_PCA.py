# !/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
created by Fangyang on Time:2018/7/18
'''

__author__ = 'Fangyang'

import numpy as np

class PCA:

    def __init__(self, n_components):
        '''初始化PCA'''
        assert n_components >= 1, 'n_components must be valid'
        self.n_components = n_components
        self.components_ = None

    def __repr__(self):
        return "PCA(n_components = %d)" % self.n_components

    def transform(self, x):
        '''将给定的x, 映射到各个主成分分量中'''
        assert x.shape[1] == self.components_.shape[1]

        return x.dot(self.components_.T)

    def inverse_transform(self, x):
        '''将给定的x, 反向映射回原来的特征空间'''
        assert x.shape[1] == self.components_.shape[0]

        return x.dot(self.components_)

    def fit(self, x, eta=0.01, n_iters=1e4):
        '''获得数据集x的前n个主成分'''
        assert self.n_components <= x.shape[1], \
            'n_components must be lower than the feature of x'

        def demean(x):
            return x - np.mean(x, axis=0)  # 在列上求mean,即 样本 - 各个特征的均值

        def f(w, x):
            '''目标函数'''
            return np.sum((x.dot(w)) ** 2) / len(x)

        def df(w, x):
            return x.T.dot(x.dot(w)) * 2. / len(x)

        def direction(w):
            '''将一个向量变成单位向量'''
            return w / np.linalg.norm(w)  # norm 求向量的模(实际上是根据ord返回范数)

        def first_component(x, initial_w, eta, n_iters=1e4, epsilon=1e-8):

            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, x)
                last_w = w  # 我们投影的时候是投影在方向向量w上面的,即需要将w变成单位方向向量
                w = w + eta * gradient
                w = direction(w)  # 注意归一化为单位方向向量 [1]
                if (abs(f(w, x) - f(last_w, x)) < epsilon):
                    break
                cur_iter += 1

            return w

        x_pca = demean(x)
        self.components_ = np.empty(shape=(self.n_components, x.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(x_pca.shape[1])
            w = first_component(x_pca, initial_w, eta, n_iters)
            self.components_[i, :] = w

            x_pca = x_pca - x_pca.dot(w).reshape(-1,1) * w

        return self