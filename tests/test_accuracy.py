import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import subprocess
import sys


def test_model_accuracy_min_threshold():

    csv_path = Path(__file__).resolve().parents[1] / "data" / "spam.csv"
    df = pd.read_csv(csv_path)
    X = df["Message"].astype(str)
    y = df["Category"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model_path = Path(__file__).resolve().parents[1] / "model.pkl"
    if not model_path.exists():
        subprocess.run([sys.executable, "train.py"], cwd=str(csv_path.parents[1]), check=True)
    model = joblib.load(model_path)

    # Si le modèle a été entraîné avec une autre version de scikit-learn,
    # on refait rapidement un fit sur X_train pour stabiliser le test
    try:
        _ = model.predict(["probe"])
    except Exception:
        import subprocess, sys
        subprocess.run([sys.executable, "train.py"], cwd=str(csv_path.parents[1]), check=True)
        model = joblib.load(model_path)

    score = model.score(X_test, y_test)
    assert score >= 0.8, f"Accuracy trop faible : {score:.4f}"


