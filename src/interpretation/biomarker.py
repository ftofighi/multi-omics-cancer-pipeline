import pandas as pd


def save_biomarkers(df, path="results/biomarkers.csv", top_n=50):
    """
    Save top SHAP-ranked genes as biomarkers.
    """

    df_top = df.head(top_n)

    df_top.to_csv(path, index=False)

    print(f"Saved top {top_n} biomarkers → {path}")

    return df_top