import numpy as np;
import pandas as pd;

class StandardScaler:
    def __init__(self):
        self.mean_ = 0
        self.scale_ = 0
    def fit(self, X):
        self.mean_ =np.mean(X,axis = 0)
        self.scale_ = np.std(X, axis = 0, ddof = 0)
        return self
    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise Exception("The data isn't fitted yet")
        else:
            X = np.array(X)
            return (X-self.mean_) / self.scale_
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
