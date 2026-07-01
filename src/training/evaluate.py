from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)

def evaluate_fold(model, X_test, y_test):

    preds = model.predict(X_test)
    probs = model.predict_proba(y_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs)
    }

    return metrics, preds, probs