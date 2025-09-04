# spam-detector

Ce projet a pour but de détecter les emails spams.

## Entraînement
```
python train.py
```

## Tests
```
python -m pytest -q
```

## Notebook
- Ouvrez `notebooks/analysis.ipynb`
- Les figures sont enregistrées dans `reports/figures`

## Streamlit (prototype minimaliste)
```
py -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Principes d'éco-conception:
- Design minimal (pas d'animations, pas d'assets lourds)
- Mise en cache des données/modèle
- Graphiques statiques compressés

## API FastAPI
```
py -m pip install -r requirements.txt
python train.py  # si model.pkl absent
py -m uvicorn api:app --reload
```

Exemple d'appel:
```
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Free entry to win cash now!\"}"
```
