import os

def test_model_exists():
    assert os.path.exists("model.pkl"), "Le modèle n'a pas été sauvegardé"

