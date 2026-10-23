"""Slice- and subject-level metrics, calibration and bootstrap confidence intervals over subjects."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def expected_calibration_error(y, prob, n_bins: int = 10) -> float:
    """ECE of the top-class confidence with equal-width bins."""
    conf, pred = prob.max(1), prob.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def classification_metrics(y, prob, n_classes: int) -> dict:
    y, prob = np.asarray(y), np.asarray(prob)
    pred = prob.argmax(1)
    out = {
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "brier": float(np.mean(np.sum((prob - np.eye(n_classes)[y]) ** 2, axis=1))),
        "ece": expected_calibration_error(y, prob),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(n_classes))).tolist(),
    }
    try:
        out["auc"] = roc_auc_score(y, prob[:, 1]) if n_classes == 2 else roc_auc_score(y, prob, multi_class="ovr")
    except ValueError:
        out["auc"] = None
    return out


def subject_level(items, prob):
    """One decision per subject: the mean of its slice probabilities. Returns (subjects, y, prob)."""
    groups = defaultdict(list)
    for it, p in zip(items, prob):
        groups[it["subject"]].append((it["y"], p))
    subjects = sorted(groups)
    ys = np.array([groups[s][0][0] for s in subjects])
    ps = np.array([np.mean([p for _, p in groups[s]], axis=0) for s in subjects])
    return subjects, ys, ps


def bootstrap_ci(y, prob, n_classes: int, metric: str = "balanced_accuracy", n: int = 1000, seed: int = 0):
    """95% percentile interval of a subject-level metric, resampling subjects."""
    rng = np.random.default_rng(seed)
    y, prob = np.asarray(y), np.asarray(prob)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        v = classification_metrics(y[idx], prob[idx], n_classes)[metric]
        if v is not None:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)
