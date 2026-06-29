import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def variance_filter(X, threshold=0.01):
    """
    Remove low-variance genes, Remove all the genes that are the same in most patients.
    """

    selector = VarianceThreshold(threshold)

    X_selected = selector.fit_transform(X)

    selected_genes = X.columns[selector.get_support()]

    X_selected = pd.DataFrame(
        X_selected,
        columns=selected_genes,
        index=X.index
    )

    print(f"Genes before selection: {X.shape[1]}")
    print(f"Genes after selection: {X_selected.shape[1]}")

    return X_selected
