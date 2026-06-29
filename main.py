from src.data.loader import load_dataset
from src.preprocessing.preprocess import preprocess
from src.feature_selection.select import variance_filter
from src.train import train_model

def main():

    print("Loading data...")
    X, y = load_dataset()

    print("Feature selection...")
    X = variance_filter(X)

    print("Preprocessing...")
    X = preprocess(X)

    print("Training model...")
    model = train_model(X, y)

    print("Done ✔")

if __name__ == "__main__":
    main()