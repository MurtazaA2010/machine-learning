import numpy  as np
import pandas as pd

class RobustScaler:
    def __init__(self):
        self.median_ = None
        self.iqr_ = None
    def fit(self, X):
        X = np.array(X)
        self.median_ = np.median(X,axis = 0)
        q1 , q3  = np.quantile(X,[0.25, 0.75], axis= 0)
        self.iqr_ = q3-q1
        return self
    def transform(self, X):
        if self.median_ is None or self.iqr_ is None:
            raise Exception("scaler not fitted yet")
        else:
            return (X- self.median_)/ self.iqr_
        
    def fit_transform(self, X):
        self.fit(X)
        return (X-self.median_) / self.iqr_