#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:15:48 2018

@author: Adekanye
"""

import numpy as np
import matplotlib.pyplot as plt
import dataGenerator


class LogReg(object):
    def __init__(self, eta=0.1, n_iter=10):
        assert eta >= 0 and eta <= 1, "eta must be between 0.0 and 1.0"
        assert type(n_iter) is int, "n_iter must be type int"
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], "X and y dimensions not compatible"
        self.coef_ = np.zeros(X.shape[1] + 1)
        for _ in range(self.n_iter):
            output = self.net_input(X)
            error = y - output
            self.coef_[0] += self.eta * error.sum()
            self.coef_[1:] += self.eta * X.T.dot(error)
        return self

    def net_input(self, X):
        return 1 / (1 + np.exp(-(X.dot(self.coef_[1:]) + self.coef_[0])))

    def predict(self, X):
        print(self.net_input(X))
        return np.where(self.net_input(X) >= 0.5, 1, 0)


#X = np.asarray([[4, 2], [3, 5], [2, 1], [7, 10], [11, 12], [10, 14]])
#y = np.asarray([1, 1, 1, 0, 0, 0])

#plt.scatter(X[0:3, 0], X[0:3, 1], color='red', marker='o', label='1')
#plt.scatter(X[3:, 0], X[3:, 1], color='blue', marker='x', label='0')
#plt.legend(loc='upper left')
#plt.show()

gen_data = dataGenerator.generate_data(x1_min=0, x2_min=0)
data, labels = gen_data.linearlySeparable(n_rows=50)
data = data.values
labels = labels.values.ravel()
log_reg = LogReg(n_iter=50)
log_reg.fit(data[0:40, :], labels[0:40,])

gen_data.plot(data[0:40, :],labels[0:40,])
