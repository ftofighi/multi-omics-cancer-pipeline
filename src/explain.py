import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def run_shap_analysis(X, y):

    os.makedirs("results", exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05
    )

    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)

    plt.savefig(
        "results/shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    importance = pd.DataFrame({
        "gene": X.columns,
        "importance": abs(shap_values).mean(axis=0)
    })

    importance = importance.sort_values(by="importance", ascending=False)

    importance.to_csv("results/top_biomarkers.csv", index=False)

    print("✔ SHAP plot saved to results/shap_summary.png")
    print("✔ Biomarkers saved to results/top_biomarkers.csv")

    return model, importance