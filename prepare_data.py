import pandas as pd

# -----------------------------
# 1. Load expression
# -----------------------------
url = "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz"
expr = pd.read_csv(url, sep="\t", index_col=0)

expr = expr.T

# -----------------------------
# 2. Create labels from sample ID
# -----------------------------
def label_from_id(sample_id):
    # TCGA barcode: TCGA-XX-XXXX-01A
    sample_type = str(sample_id).split("-")[3][:2]
    
    if sample_type == "01":
        return 1  # tumor
    else:
        return 0  # normal

labels = pd.DataFrame(index=expr.index)
labels["label"] = [label_from_id(i) for i in expr.index]

# -----------------------------
# 3. Save
# -----------------------------
expr.to_csv("data/expression.csv")
labels.to_csv("data/labels.csv")

print("✔ Data ready:", expr.shape, labels.shape)