"""Compare the slice-level and subject-level split experiments in a single table.

Example:
    python src/report.py --runs runs/slice runs/subject
"""

import argparse
import json
import os


def fmt(v):
    return "-" if v is None else f"{v:.3f}"


def main():
    p = argparse.ArgumentParser(description="Experiment comparison table")
    p.add_argument("--runs", nargs="+", default=["runs/slice", "runs/subject"])
    p.add_argument("--out", default="runs/report.md")
    a = p.parse_args()

    names = {"slice": "Slice-level split (leaky)", "subject": "Subject-level split"}
    lines = ["| Split | Train ∩ test subjects | Slice accuracy | Slice macro-F1 | Subject accuracy | Subject macro-F1 | Subject AUC |",
             "|---|---|---|---|---|---|---|"]
    for r in a.runs:
        m = json.load(open(os.path.join(r, "metrics.json"), encoding="utf-8"))
        s, h = m["slice_level"], m["subject_level"]
        lines.append(f"| {names.get(m['split'], m['split'])} | {m['train_test_subject_overlap']} | {fmt(s['accuracy'])} | "
                     f"{fmt(s['macro_f1'])} | {fmt(h['accuracy'])} | {fmt(h['macro_f1'])} | {fmt(h['auc'])} |")
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(text + "\n")
    print(text + f"\n\nSaved: {a.out}")


if __name__ == "__main__":
    main()
