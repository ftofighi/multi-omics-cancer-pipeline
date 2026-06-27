from src.data.loader import load_dataset
from src.preprocessing.preprocess import preprocess
from src.train import train_model

def main():

    print("Loading data...")
    X, y = load_dataset()

    print("Preprocessing...")
    X = preprocess(X)

    print("Training model...")
    model = train_model(X, y)

    print("Done ✔")

if __name__ == "__main__":
    main()