"""K-fold cross-validation of both split protocols, with and without permuted labels.

Every configuration is run on K folds with the same seeds:

* ``subject`` folds (StratifiedGroupKFold): each subject is tested exactly once, by a model that
  never saw it. The out-of-fold subject predictions are pooled into one estimate with a bootstrap CI.
* ``slice`` folds: the leaky protocol, to measure how much it inflates the result.
* ``--permute`` adds both protocols with labels shuffled between subjects. With subject folds this
  must give chance-level accuracy; with slice folds anything above chance is subject memorisation.

    python -m admri.cv --root <data folder> --folds 5 --epochs 8 --permute --out runs/cv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import assign_labels, cv_folds, permute_subject_labels, scan, subsample
from .metrics import bootstrap_ci, classification_metrics
from .train import fit_and_evaluate
from .train import parse_args as train_args


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Cross-validated slice- vs subject-level comparison")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--protocols", nargs="+", default=["slice", "subject"], choices=["slice", "subject"])
    p.add_argument("--permute", action="store_true", help="Also run with labels permuted between subjects")
    p.add_argument("--out", default="runs/cv")
    a, rest = p.parse_known_args(argv)
    t = train_args(rest + ["--out", a.out])
    return a, t


def summarise(df: pd.DataFrame, pooled: dict) -> str:
    lines = ["| Protocol | Labels | Slice bal. acc | Subject bal. acc | Subject AUC | Subject ECE | "
             "Pooled subject bal. acc (95% CI) |", "|---|---|---|---|---|---|---|"]
    for (proto, labels), g in df.groupby(["protocol", "labels"], sort=False):
        p = pooled[f"{proto}/{labels}"]
        lo, hi = p["balanced_accuracy_ci"]
        lines.append(f"| {proto} | {labels} | {g.slice_bacc.mean():.3f} ± {g.slice_bacc.std():.3f} | "
                     f"{g.subject_bacc.mean():.3f} ± {g.subject_bacc.std():.3f} | "
                     f"{g.subject_auc.mean():.3f} ± {g.subject_auc.std():.3f} | {g.subject_ece.mean():.3f} | "
                     f"{p['balanced_accuracy']:.3f} ({lo:.3f}–{hi:.3f}) |")
    true = df[df.labels == "true"].pivot(index="fold", columns="protocol", values="subject_bacc")
    if {"slice", "subject"} <= set(true.columns):
        gap = true["slice"] - true["subject"]
        lines += ["", f"Leakage gap (slice − subject, subject-level balanced accuracy, paired by fold): "
                      f"{gap.mean():+.3f} ± {gap.std():.3f} over {len(gap)} folds."]
    lines += ["", "Mean ± std over folds. The pooled estimate scores all out-of-fold subject predictions together; "
                  "its interval resamples subjects."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    a, t = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items, classes = assign_labels(scan(t.root, t.subject_regex), t.classes)
    items = subsample(items, t.slices_per_subject)
    n = len(classes)
    label_sets = {"true": items}
    if a.permute:
        label_sets["permuted"] = permute_subject_labels(items, t.seed)

    rows, pooled = [], {}
    for labels, its in label_sets.items():
        for proto in a.protocols:
            ys, ps = [], []
            for fold, (tr, va, te) in enumerate(cv_folds(its, a.folds, proto, t.seed)):
                print(f"\n=== {proto} split, {labels} labels, fold {fold + 1}/{a.folds} ===")
                r = fit_and_evaluate(tr, va, te, classes, t, log=lambda *_: None)
                s, h = r["slice_level"], r["subject_level"]
                rows.append({"protocol": proto, "labels": labels, "fold": fold,
                             "overlap": r["train_test_subject_overlap"],
                             "slice_bacc": s["balanced_accuracy"], "subject_bacc": h["balanced_accuracy"],
                             "subject_auc": h["auc"], "subject_ece": h["ece"], "test_subjects": r["test_subjects"]})
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
                if proto == "subject":  # each subject appears in exactly one test fold
                    ys += [p["y"] for p in r["subject_predictions"]]
                    ps += [p["p"] for p in r["subject_predictions"]]
                else:  # slice folds can test a subject several times; average its predictions
                    for p in r["subject_predictions"]:
                        ys.append(p["y"])
                        ps.append(p["p"])
            ys, ps = np.array(ys), np.array(ps)
            m = classification_metrics(ys, ps, n)
            m["balanced_accuracy_ci"] = bootstrap_ci(ys, ps, n, "balanced_accuracy", t.bootstrap)
            pooled[f"{proto}/{labels}"] = m

    df = pd.DataFrame(rows)
    df.to_csv(out / "folds.csv", index=False)
    text = summarise(df, pooled)
    (out / "summary.md").write_text(text, encoding="utf-8")
    (out / "pooled.json").write_text(json.dumps(pooled, indent=2, default=float))
    print("\n" + text)


if __name__ == "__main__":
    main()
