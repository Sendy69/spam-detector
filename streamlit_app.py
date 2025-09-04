import os
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from utils_text import clean_texts
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "spam.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
FIG_DIR = BASE_DIR / "reports" / "figures"

st.set_page_config(page_title="Analyse Spam - Régression Logistique", layout="centered")

st.title("Analyse Spam - Régression Logistique")
st.caption("Interface minimaliste, centrée sur l'essentiel (éco-conception)")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = clean_texts(df["Message"].astype(str))
    y = df["Category"].astype(str)
    return X, y

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(BASE_DIR / "train.py")], check=True)
    return joblib.load(MODEL_PATH)

X, y = load_data()
model = load_model()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Métriques clés")
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{acc:.2%}")
with col2:
    st.write("")

st.subheader("Graphiques d'évaluation")

def show_image(name: str, caption: str):
    path = FIG_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"{caption} (placeholder)")

show_image("confusion_matrix.png", "Matrice de confusion")
show_image("roc_curve.png", "Courbe ROC")
show_image("pr_curve.png", "Courbe Précision-Rappel")
show_image("calibration_curve.png", "Courbe de calibration")

st.divider()
st.subheader("Analyse d'un message")

threshold = st.slider("Seuil de décision (probabilité spam)", 0.0, 1.0, 0.50, 0.01)
user_text = st.text_area("Collez un message à analyser", height=120)
if st.button("Analyser") and user_text.strip():
    cleaned = clean_texts([user_text])[0]
    pred = model.predict([cleaned])[0]
    proba = None
    try:
        classes = list(getattr(model, "classes_", []))
        if classes and "spam" in classes:
            pos_idx = classes.index("spam")
            proba = float(model.predict_proba([cleaned])[0][pos_idx])
    except Exception:
        proba = None

    final_label = pred
    if proba is not None:
        final_label = "spam" if proba >= threshold else "ham"

    st.write(f"Prédiction: **{final_label}**")
    if proba is not None:
        st.write(f"Probabilité spam: **{proba:.2%}** (seuil {threshold:.0%})")

st.write("""
Conseils d'éco-conception appliqués:
- Design minimaliste, pas d'images lourdes ni d'animations
- Mise en cache des données et du modèle
- Graphiques statiques uniquement quand nécessaires
""")


