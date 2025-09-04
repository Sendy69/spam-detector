import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# chargement du dataset 
def train_spam_model(csv_path="data/spam.csv", test_size=0.2, random_state=42, max_features=5000):
    """
    Charge les données, prétraite les messages, entraîne un modèle Logistic Regression,
    et retourne le modèle, le vectorizer et les métriques.
    """
    # Charger les données
    data = pd.read_csv(csv_path)
    
    # Nettoyage simple du texte
    data['Message'] = (
        data['Message']
        .str.lower()
        .str.replace(r'[^a-z\s]', '', regex=True)
        .str.strip()
    )
    
    # Séparer features et labels
    X = data["Message"]
    y = data["Category"]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Entraînement du modèle
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test_vec)
    
    # Métriques
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)
    
    # Retourner tout ce qui peut être utile
    return model, vectorizer, report, X_test_vec, y_test, y_pred

model, vectorizer, report, X_test_vec, y_test, y_pred= train_spam_model("data/spam.csv")

# sauvegarde du modèle et du vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Modèle entraîné et sauvegardé")

