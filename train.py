import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils_text import clean_texts


def load_dataset(csv_path: str) -> tuple[pd.Series, pd.Series]:

    df = pd.read_csv(csv_path)

    if "Message" not in df.columns or "Category" not in df.columns:
        raise ValueError("Le dataset doit contenir les colonnes 'Category' et 'Message'.")

    X = clean_texts(df["Message"].astype(str))
    y = df["Category"].astype(str)
    return X, y


def build_pipeline() -> Pipeline:

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=50000,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=None)),
        ]
    )
    return pipeline


def train_and_save(
    csv_path: str = "data/spam.csv",
    model_path: str = "model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:

    X, y = load_dataset(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, model_path)
    print(f"Modèle sauvegardé dans {model_path}")


if __name__ == "__main__":
    train_and_save()


