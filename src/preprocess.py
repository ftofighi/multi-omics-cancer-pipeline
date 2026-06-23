import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data():
    X = pd.read_csv("data/expression.csv", index_col=0)
    y = pd.read_csv("data/labels.csv", index_col=0)

    # align
    X = X.loc[y.index]

    return X, y.values.ravel()


def preprocess(X):
    # log transform
    X = np.log1p(X)

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled