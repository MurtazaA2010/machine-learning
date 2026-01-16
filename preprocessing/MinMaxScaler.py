import numpy as np
import pandas as pd

class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None
    def fit(self, X):
        X = np.array(X)
        self.min_ = np.min(X)
        self.max_ = np.max(X)
        return self
    def transform(self, X):
        if self.min_ is None or self.max_ is None:
            raise Exception("min max scaller not fitted yet!!")
        else:
            if (self.max_ == self.min_):
                return (X-self.min_) / ((self.max_ - self.min_) + 1e-9)  ##addint 1e-9 to make sure the denominator never equals to zero
            else:
                return (X-self.min_) / ((self.max_ - self.min_))  
    def fit_transform(self, X):
        self.fit(X)
        if (self.max_ == self.min_):
            return (X-self.min_) / ((self.max_ - self.min_) + 1e-9)  
        else:
            return (X-self.min_) / ((self.max_ - self.min_))