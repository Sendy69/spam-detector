import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from train import train_spam_model  # ta fonction d'entraînement

st.set_page_config(page_title="Spam Detector - Résultats", layout="centered")
st.title("Analyse Spam Detector")

# --- Option pour réentraîner ou utiliser modèle existant ---
use_saved = st.checkbox("Utiliser le modèle sauvegardé si disponible", value=True)

if use_saved:
    try:
        # Charger modèle et vectorizer existants
        model_use = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        st.success("Modèle et vectorizer chargés depuis les fichiers sauvegardés.")
    except:
        st.warning("Fichiers sauvegardés introuvables, entraînement du modèle en cours...")
        model, vectorizer, report, cm = train_spam_model("data/spam.csv")
        joblib.dump(model, "model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
else:
    model, vectorizer, report, cm = train_spam_model("data/spam.csv")
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    st.success("Modèle entraîné et sauvegardé.")

# --- Charger les données pour les graphiques ---
df = pd.read_csv("data/spam.csv")
df['Message'] = df['Message'].str.lower().str.replace(r'[^a-z\s]', '', regex=True).str.strip()

# Vectorisation pour prédictions et métriques
X_vec = vectorizer.transform(df["Message"])
y_true = df["Category"]
y_pred = model_use.predict(X_vec)

# --- 1️⃣ Distribution de la longueur des messages ---
df["length"] = df["Message"].apply(len)
fig1, ax1 = plt.subplots()
sns.histplot(df, x="length", hue="Category", bins=30, kde=False,
             palette=["#1f77b4","#ff7f0e"], ax=ax1)
ax1.set_title("Distribution de la longueur des emails")
ax1.set_xlabel("Nombre de caractères")
ax1.set_ylabel("Nombre d'emails")
st.pyplot(fig1)

# --- 2️⃣ Matrice de confusion ---
model, vectorizer, report, X_test, y_test, y_pred = train_spam_model("data/spam.csv")

cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_, ax=ax2)
ax2.set_xlabel("Prédit")
ax2.set_ylabel("Réel")
ax2.set_title("Matrice de confusion (jeu de test)")
st.pyplot(fig2)


# --- 3️⃣ Courbe ROC (si binaire) ---
# Courbe ROC uniquement sur test
fig3, ax3 = plt.subplots()
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax3)
ax3.set_title("Courbe ROC (jeu de test)")
st.pyplot(fig3)

# --- 4️⃣ Importance des mots ---
coef = pd.Series(model.coef_[0], index=vectorizer.get_feature_names_out())
top_positive = coef.sort_values(ascending=False).head(10)
top_negative = coef.sort_values().head(10)

fig4, ax4 = plt.subplots(figsize=(6,4))
top_positive.plot(kind="barh", color="#ff7f0e", ax=ax4)
ax4.set_title("Mots les plus associés au spam")
st.pyplot(fig4)

fig5, ax5 = plt.subplots(figsize=(6,4))
top_negative.plot(kind="barh", color="#1f77b4", ax=ax5)
ax5.set_title("Mots les plus associés au non-spam")
st.pyplot(fig5)
