"""End-to-end on synthetic slices: the leakage probe must separate the two split protocols."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "2"}
FAST = ["--model", "smallcnn", "--img-size", "64", "--epochs", "10", "--batch", "32", "--workers", "0",
        "--lr", "3e-3", "--bootstrap", "100", "--seed", "0"]


def run(*args):
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=ROOT, env=ENV)


def test_train_report_gradcam(tmp_path):
    data = tmp_path / "data"
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic_oasis.py"), "--out", str(data),
                    "--subjects", "24", "--slices", "8"], check=True)
    for split in ("slice", "subject"):
        run("admri.train", "--root", str(data), "--split", split, "--out", str(tmp_path / split), *FAST)
    m = json.loads((tmp_path / "subject" / "metrics.json").read_text())
    assert m["train_test_subject_overlap"] == 0
    assert json.loads((tmp_path / "slice" / "metrics.json").read_text())["train_test_subject_overlap"] > 0
    run("admri.report", "--runs", str(tmp_path / "slice"), str(tmp_path / "subject"), "--out", str(tmp_path / "r.md"))
    run("admri.gradcam", "--run", str(tmp_path / "subject"), "--n", "2", "--all")
    assert (tmp_path / "subject" / "gradcam.png").exists()
    assert 0 <= json.loads((tmp_path / "subject" / "gradcam_stats.json").read_text())["mean_energy_in_brain"] <= 1


def test_cv_leakage_probe(tmp_path):
    """No disease signal at all: only a model that has seen the test subjects can beat chance."""
    data = tmp_path / "data"
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic_oasis.py"), "--out", str(data),
                    "--subjects", "30", "--slices", "10", "--signal", "0"], check=True)
    run("admri.cv", "--root", str(data), "--folds", "3", "--out", str(tmp_path / "cv"), *FAST)
    df = pd.read_csv(tmp_path / "cv" / "folds.csv")
    by = df.groupby("protocol")["subject_bacc"].mean()
    assert (df[df.protocol == "subject"]["overlap"] == 0).all()
    assert by["slice"] > by["subject"] + 0.1
    assert "Leakage gap" in (tmp_path / "cv" / "summary.md").read_text()
