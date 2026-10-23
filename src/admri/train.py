"""Dementia classification from MRI slices: training and evaluation with slice-level vs subject-level splits.

The same model and settings are trained with two different splits:

  --split slice   : slices are split at random (common but flawed: neighbouring slices of the same
                    subject end up in both train and test -> data leakage)
  --split subject : subjects are split; test subjects are never seen during training

Test results are reported per slice and per subject (mean of a subject's slice probabilities), with
a 95% bootstrap interval over subjects. ``--permute-labels`` shuffles labels between subjects: any
accuracy above chance is then pure subject memorisation, i.e. a direct measurement of leakage.

Example::

    python -m admri.train --root <data folder> --split subject --epochs 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .data import SliceDataset, assign_labels, describe, permute_subject_labels, scan, split_items, subsample
from .metrics import bootstrap_ci, classification_metrics, subject_level
from .models import MEAN, STD, build_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Alzheimer MRI classification with slice- vs subject-level splits")
    p.add_argument("--root", required=True, help="Data folder containing the class folders")
    p.add_argument("--split", choices=["slice", "subject"], default="subject")
    p.add_argument("--classes", choices=["binary", "four"], default="binary")
    p.add_argument("--subject-regex", default=r"(OAS1_\d{4})", help="Regex that extracts the subject ID")
    p.add_argument("--slices-per-subject", type=int, default=20)
    p.add_argument("--permute-labels", action="store_true", help="Shuffle labels between subjects (leakage probe)")
    p.add_argument("--model", default="resnet18", choices=["resnet18", "resnet18_scratch", "smallcnn"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--out", default=None, help="Default: runs/<split>")
    return p.parse_args(argv)


def transforms_for(img_size: int):
    train = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(7), transforms.ColorJitter(0.1, 0.1),
                                transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ev = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    return train, ev


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probs.append(torch.softmax(model(x.to(device)).float(), dim=1).cpu())
    return torch.cat(probs).numpy()


def fit_and_evaluate(train, val, test, classes, a, out=None, log=print) -> dict:
    """Train on ``train``, select the epoch on ``val`` (subject-level balanced accuracy), evaluate on ``test``."""
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf_train, tf_eval = transforms_for(a.img_size)

    def dl(part, tf, shuffle):
        return DataLoader(SliceDataset(part, tf), batch_size=a.batch, shuffle=shuffle, num_workers=a.workers,
                          pin_memory=device.type == "cuda")

    train_dl, val_dl, test_dl = dl(train, tf_train, True), dl(val, tf_eval, False), dl(test, tf_eval, False)
    n = len(classes)
    counts = np.bincount([it["y"] for it in train], minlength=n).astype(float)
    weights = torch.tensor(counts.sum() / (n * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model = build_model(a.model, n).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)  # class-weighted loss for class imbalance
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best, best_state, history, t0 = -1.0, None, [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
        sched.step()
        _, vy, vp = subject_level(val, predict(model, val_dl, device))
        val_bacc = classification_metrics(vy, vp, n)["balanced_accuracy"]
        history.append({"epoch": ep, "train_loss": total / len(train), "val_subject_bacc": val_bacc})
        log(f"Epoch {ep}/{a.epochs} | train loss {total / len(train):.4f} | val subject bal. acc {val_bacc:.4f}")
        if val_bacc > best:  # model selection uses the validation subjects, never the test set
            best = val_bacc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if out:
        torch.save(best_state, os.path.join(out, "model.pt"))

    prob = predict(model, test_dl, device)
    y_test = np.array([it["y"] for it in test])
    subjects, ys, ps = subject_level(test, prob)
    subj = classification_metrics(ys, ps, n)
    subj["balanced_accuracy_ci"] = bootstrap_ci(ys, ps, n, "balanced_accuracy", a.bootstrap) if a.bootstrap else None
    return {
        "classes": classes,
        "train_test_subject_overlap": len({it["subject"] for it in train} & {it["subject"] for it in test}),
        "test_slices": len(test),
        "test_subjects": len(ys),
        "slice_level": classification_metrics(y_test, prob, n),
        "subject_level": subj,
        "history": history,
        "runtime_min": round((time.time() - t0) / 60, 2),
        "subject_predictions": [{"subject": s, "y": int(y), "p": p.tolist()} for s, y, p in zip(subjects, ys, ps)],
    }


def main(argv=None):
    a = parse_args(argv)
    out = a.out or os.path.join("runs", a.split + ("_permuted" if a.permute_labels else ""))
    os.makedirs(out, exist_ok=True)

    items, classes = assign_labels(scan(a.root, a.subject_regex), a.classes)
    items = subsample(items, a.slices_per_subject)
    if a.permute_labels:
        items = permute_subject_labels(items, a.seed)
    train, val, test = split_items(items, a.split, a.seed)
    note = " | labels permuted between subjects" if a.permute_labels else ""
    print(f"Split: {a.split} | Classes: {classes}{note}")
    describe(train, val, test, classes)

    result = {"split": a.split, "permuted_labels": a.permute_labels, "args": vars(a),
              **fit_and_evaluate(train, val, test, classes, a, out)}
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    with open(os.path.join(out, "split.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "subject", "y", "split"])
        w.writeheader()
        for it in train + val + test:
            w.writerow({k: it[k] for k in ("path", "subject", "y", "split")})

    s, h = result["slice_level"], result["subject_level"]
    lo, hi = h["balanced_accuracy_ci"] or (float("nan"), float("nan"))
    print(f"\n=== Test ({a.split} split) ===")
    print(f"Slice level   : accuracy {s['accuracy']:.3f} | bal. acc {s['balanced_accuracy']:.3f} | AUC {s['auc']}")
    print(f"Subject level : accuracy {h['accuracy']:.3f} | bal. acc {h['balanced_accuracy']:.3f} "
          f"[{lo:.3f}, {hi:.3f}] | AUC {h['auc']} | ECE {h['ece']:.3f}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
