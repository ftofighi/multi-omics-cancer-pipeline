from xgboost import XGBClassifier

def build_model():

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )