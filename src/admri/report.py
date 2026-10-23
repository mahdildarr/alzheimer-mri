"""Compare the slice-level and subject-level split experiments in a single table.

Example::

    python -m admri.report --runs runs/slice runs/subject runs/slice_permuted runs/subject_permuted
"""
from __future__ import annotations

import argparse
import json
import os


def fmt(v):
    return "-" if v is None else f"{v:.3f}"


def main(argv=None):
    p = argparse.ArgumentParser(description="Experiment comparison table")
    p.add_argument("--runs", nargs="+", default=["runs/slice", "runs/subject"])
    p.add_argument("--out", default="runs/report.md")
    a = p.parse_args(argv)

    names = {"slice": "Slice-level split (leaky)", "subject": "Subject-level split"}
    lines = ["| Split | Labels | Train ∩ test subjects | Slice accuracy | Subject bal. acc (95% CI) | Subject AUC | "
             "Subject ECE |", "|---|---|---|---|---|---|---|"]
    for r in a.runs:
        m = json.load(open(os.path.join(r, "metrics.json"), encoding="utf-8"))
        s, h = m["slice_level"], m["subject_level"]
        ci = h.get("balanced_accuracy_ci") or (None, None)
        labels = "permuted" if m.get("permuted_labels") else "true"
        lines.append(f"| {names.get(m['split'], m['split'])} | {labels} | {m['train_test_subject_overlap']} | "
                     f"{fmt(s['accuracy'])} | {fmt(h['balanced_accuracy'])} ({fmt(ci[0])}–{fmt(ci[1])}) | "
                     f"{fmt(h['auc'])} | {fmt(h.get('ece'))} |")
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(text + "\n")
    print(text + f"\n\nSaved: {a.out}")


if __name__ == "__main__":
    main()
