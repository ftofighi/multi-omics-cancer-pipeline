import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def log_transform(X):
    return np.log1p(X)


def standardize(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )


def preprocess(X):
    X = log_transform(X)
    X = standardize(X)
    return X