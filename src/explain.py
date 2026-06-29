import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def run_shap_analysis(X, y):

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training model...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05
    )

    model.fit(X_train, y_train)

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame({
        "gene": X.columns,
        "importance": abs(shap_values).mean(axis=0)
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False
    )

    importance.to_csv("top_biomarkers.csv", index=False)

    print("Top biomarkers:")
    print(importance.head(10))

    return model, importance