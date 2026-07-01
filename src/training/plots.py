from sklearn.metrics import (
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    confusion_matrix
)

import matplotlib.pyplot as plt

def save_roc(y_true, probs, save_path):
    RocCurveDisplay.from_predictions(
        y_true,
        probs
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

def save_confusion_matrix(y_true, preds, save_path):
    cm = confusion_matrix(y_true, preds)

    ConfusionMatrixDisplay(cm).plot()

    plt.savefig(save_path, dpi=300)
    plt.close()