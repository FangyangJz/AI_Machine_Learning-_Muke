# !/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
created by Fangyang on Time:2018/7/19
'''

__author__ = 'Fangyang'
# 200+MB那个 老老实实用迅雷下吧 https://ndownloader.figshare.com/files/5976015

# import numpy as np
# import matplotlib.pyplot as plt
# import os
from sklearn.datasets import fetch_lfw_people
#
# path = os.getcwd()
faces = fetch_lfw_people(data_home=path)