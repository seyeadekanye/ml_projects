#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:27:23 2018

@author: Adekanye
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class generate_data(object):
    def __init__(self, x1_min=-100, x1_max=100, x2_min=-100, x2_max=100,
                 step=0.1):
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_min = x2_min
        self.x2_max = x2_max
        self.step = step

    def linearlySeparable(self, n_rows=10, n_cols=2):
        assert type(n_rows) is int
        assert type(n_cols) is int
        X = pd.DataFrame(np.zeros([n_rows, n_cols]))
        y = pd.DataFrame(np.zeros([n_rows, ]))
        X1 = np.arange(self.x1_min, self.x1_max, self.step)
        X2 = np.arange(self.x2_min, self.x2_max, self.step)
        for i in range(n_rows):
            xx1 = np.random.choice(X1, 1)
            xx2 = np.random.choice(X2, 1)
            X.iloc[[i], 0] = xx1
            X.iloc[[i], 1] = xx2
            if xx1 <= xx2:
                y.iloc[[i], ] = 1
            else:
                y.iloc[[i], ] = 0
        return X, y

    def plot(self, data, labels):
        data = pd.DataFrame(data)
        labels = pd.DataFrame(labels)
        positive_class_index = labels[labels[0] == 1].index
        negative_class_index = labels[labels[0] == 0].index
        plt.scatter(data.iloc[positive_class_index][0],
                    data.iloc[positive_class_index][1],
                    color='red', marker='o', label='1')
        plt.scatter(data.iloc[negative_class_index][0],
                    data.iloc[negative_class_index][1],
                    color='blue', marker='x', label='0')
        plt.legend(loc='upper left')
        plt.show()


gen_data = generate_data(x1_min=-100, x2_min=-100)
data, labels = gen_data.linearlySeparable(n_rows=10)
gen_data.plot(data,labels)