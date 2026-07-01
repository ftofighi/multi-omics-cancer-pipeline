import numpy as np

from sklearn.model_selection import StratifiedKFold
from src.models.xgboost_model import build_model
from src.training.evaluate import evaluate_fold

def train_model(X, y):

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    accs = []
    aucs = []

    fold_results = []

    last = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = build_model()

        model.fit(X_train, y_train)

        metrics, preds, probs = evaluate_fold(
            model,
            X_test,
            y_test
        )

        accs.append(metrics["accuracy"])
        aucs.append(metrics["auc"])

        fold_results.append({
            "Fold": fold,
            "Accuracy": metrics["accuracy"],
            "AUC": metrics["auc"]
        })

        last = {
           "y_true": y_test,
           "preds": preds,
           "probs": probs 
        }
    
    summary = {
        "accuracy_mean": np.mean(accs),
        "accuracy_std": np.std(accs),
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs)
    }

    return model, fold_results, summary, last
