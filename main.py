from src.train import train_model
from src.data.loader import load_dataset
from src.preprocessing.preprocess import preprocess
from src.feature_selection.select import variance_filter
from src.feature_selection.supervised import anova_feature_selection
from src.interpretation.shap_explainer import compute_shap_values, get_shap_importance
from src.interpretation.biomarker import save_biomarkers
from src.explain import run_shap_analysis


def main():

    print("Loading data...")
    X, y = load_dataset()

    print("Feature selection...")
    X = variance_filter(X)

    print("Supervised feature selection (ANOVA)...")
    X, selector = anova_feature_selection(X, y, k=1000)

    print("Preprocessing...")
    X = preprocess(X)

    print("Training model...")
    model = train_model(X, y)

    print("SHAP + Biomarkers...")
    model, importance = run_shap_analysis(X, y)

    print("Done ✔")

if __name__ == "__main__":
    main()