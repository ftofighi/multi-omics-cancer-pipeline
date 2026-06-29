from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


def train_model(X, y):

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss"
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs = []
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        accs.append(acc)
        aucs.append(auc)

        print(f"Fold {fold+1}: ACC={acc:.4f}, AUC={auc:.4f}")

    print("\nFinal Results")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"AUC     : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    return model