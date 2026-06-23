from src.preprocess import load_data, preprocess
from src.train import train_model

def main():

    print("Loading data...")
    X, y = load_data()

    print("Preprocessing...")
    X = preprocess(X)

    print("Training model...")
    model = train_model(X, y)

    print("Done ✔")

if __name__ == "__main__":
    main()