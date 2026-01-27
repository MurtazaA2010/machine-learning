import numpy as np 
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate, n_iters) :
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None    
    def fit(self, X,y ):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1 / X.shape[0]) * np.sum(y_pred - y)      
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self.weights, self.bias 

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias     

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.predict(X)