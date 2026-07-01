from pathlib import Path

from src.data.loader import load_dataset
from src.preprocessing.preprocess import preprocess
from src.feature_selection.select import variance_filter
from src.feature_selection.supervised import anova_feature_selection

from src.training.train import train_model
from src.explain import run_shap_analysis

from src.training.save_results import (
    save_metrics,
    save_fold_results
)

from src.training.plots import (
    save_roc,
    save_confusion_matrix
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():

    # ----------------------------------------------------
    # Load dataset
    # ----------------------------------------------------
    print("Loading dataset...")
    X, y = load_dataset()

    # ----------------------------------------------------
    # Feature selection
    # ----------------------------------------------------
    print("Variance filtering...")
    X = variance_filter(X)

    print("ANOVA feature selection...")
    X, selector = anova_feature_selection(
        X,
        y,
        k=1000
    )

    # ----------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------
    print("Preprocessing...")
    X = preprocess(X)

    # ----------------------------------------------------
    # Model Training
    # ----------------------------------------------------
    print("Training XGBoost model...")

    model, fold_results, metrics, last = train_model(
        X,
        y
    )

    # ----------------------------------------------------
    # Save evaluation
    # ----------------------------------------------------
    print("Saving evaluation results...")

    save_fold_results(
        fold_results,
        RESULTS_DIR / "fold_results.csv"
    )

    save_metrics(
        metrics,
        RESULTS_DIR / "metrics.json"
    )

    save_roc(
        last["y_test"],
        last["probs"],
        RESULTS_DIR / "roc_curve.png"
    )

    save_confusion_matrix(
        last["y_test"],
        last["preds"],
        RESULTS_DIR / "confusion_matrix.png"
    )

    # ----------------------------------------------------
    # SHAP Interpretation
    # ----------------------------------------------------
    print("Running SHAP analysis...")

    run_shap_analysis(
        X,
        y
    )

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()