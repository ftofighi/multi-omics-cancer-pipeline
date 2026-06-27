import numpy as np
from sklearn.preprocessing import StandardScaler

def log_transform(X):

    return np.log1p(X)

def standardize(X):

    scaler = StandardScaler()

    return scaler.fit_transform(X)

def preprocess(X):

    X = log_transform(X)
    X = standardize(X)

    return X

