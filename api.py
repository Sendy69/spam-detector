from pathlib import Path
from typing import Optional
from functools import lru_cache

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils_text import clean_text


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

app = FastAPI(title="Spam Detector API", version="1.0.0")


class AnalyzeRequest(BaseModel):
    message: str = Field(..., description="Raw email message text")


class AnalyzeResponse(BaseModel):
    label: str
    probability_spam: Optional[float] = None


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        # Essayer d'entraîner si manquant
        try:
            import subprocess, sys
            subprocess.run([sys.executable, str(BASE_DIR / "train.py")], check=True)
        except Exception:
            raise FileNotFoundError("model.pkl introuvable et entraînement automatique échoué.")
    return joblib.load(MODEL_PATH)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        text = clean_text(req.message)
        model = get_model()
        pred = model.predict([text])[0]
        proba = None
        try:
            # Chercher index de la classe 'spam'
            classes = list(getattr(model, "classes_", []))
            if classes and "spam" in classes:
                pos_idx = classes.index("spam")
                proba = float(model.predict_proba([text])[0][pos_idx])
        except Exception:
            proba = None
        return AnalyzeResponse(label=str(pred), probability_spam=proba)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse: {e}")


