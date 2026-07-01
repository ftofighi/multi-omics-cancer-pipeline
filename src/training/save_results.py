import json
import pandas as pd

def save_metrics(metrics, path):

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

def save_fold_results(results, path):

    pd.DataFrame(results).to_csv(
        path,
        index=False
    )