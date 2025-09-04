import os
import sys
import subprocess
from pathlib import Path


def test_model_exists_after_training():

    project_dir = Path(__file__).resolve().parents[1]
    model_path = project_dir / "model.pkl"

    if model_path.exists():
        model_path.unlink()

    subprocess.run([sys.executable, "train.py"], cwd=str(project_dir), check=True)

    assert model_path.exists()
    assert model_path.stat().st_size > 0


