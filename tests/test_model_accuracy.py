import pandas as pd
import joblib

def test_model_accuracy():
    df = pd.read_csv("data/spam.csv")
    X, y = df["Message"], df["Category"]
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    X_vec = vectorizer.transform(X)
    score = model.score(X_vec, y)
    assert score >= 0.8, f"Accuracy trop faible : {score}"