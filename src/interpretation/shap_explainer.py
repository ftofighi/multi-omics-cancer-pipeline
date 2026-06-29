import shap
import pandas as pd


def compute_shap_values(model, X):
    """
    Compute SHAP values for trained XGBoost model.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    return shap_values


def get_shap_importance(shap_values, X):
    """
    Convert SHAP values into gene importance ranking.
    """

    importance = abs(shap_values).mean(axis=0)

    df = pd.DataFrame({
        "gene": X.columns,
        "shap_importance": importance
    })

    df = df.sort_values("shap_importance", ascending=False)

    return df