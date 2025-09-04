import os
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(scope="session")
def client():
    return TestClient(api.app)


def test_analyze_endpoint_smoke(client):
    # Ensure model exists
    model_path = Path(__file__).resolve().parents[1] / "model.pkl"
    if not model_path.exists():
        import subprocess, sys
        subprocess.run([sys.executable, "train.py"], cwd=str(model_path.parents[0]), check=True)

    resp = client.post("/analyze", json={"message": "Free entry to win cash now!"})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"label", "probability_spam"}
    assert data["label"] in {"spam", "ham"}
    if data["probability_spam"] is not None:
        assert 0.0 <= data["probability_spam"] <= 1.0


