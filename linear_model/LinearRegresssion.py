import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        m, n = X.shape
        k = y.shape[1]

        self.weights = np.zeros((n, k))
        self.bias = np.zeros((1, k))

        for _ in range(self.n_iters):
            y_pred = X.dot(self.weights) + self.bias  # (m x k)
            dw = (1 / m) * X.T.dot(y_pred - y)        # (n x k)
            db = (1 / m) * np.sum(y_pred - y, axis=0, keepdims=True)  # (1 x k)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self.weights, self.bias

    def predict(self, X):
        y_pred = X.dot(self.weights) + self.bias
        # If single output, return 1D array
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.predict(X)
