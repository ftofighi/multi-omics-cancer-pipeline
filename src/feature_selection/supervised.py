import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def anova_feature_selection(X,y, k=1000):
    """
    Select top-k genes using ANOVA F-test.
    This is a supervised feature selection method.
    """

    selector = SelectKBest(score_func=f_classif, k=k)

    X_selected = selector.fit_transform(X, y)

    selected_genes = X.columns[selector.get_support()]

    X_selected = pd.DataFrame(
        X_selected,
        columns=selected_genes,
        index=X.index
    )

    print(f"Anove Genes before: {X.shape[1]}")
    print(f"Anova Genes after: {X_selected.shape[1]}")

    return X_selected, selector


    