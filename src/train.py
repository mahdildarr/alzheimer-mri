"""Dementia classification from MRI slices: training and evaluation with slice-level vs. subject-level splits.

The same model and settings are trained with two different splits:
  --split slice   : slices are split at random (a common but flawed approach; neighboring slices of the same
                    subject end up in both train and test → data leakage)
  --split subject : subjects are split; test subjects are never seen during training (the correct approach)
Test results are reported both per slice and per subject (mean of all slice probabilities of a subject).

Example:
    python src/train.py --root <data folder> --split subject --epochs 8
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models, transforms

from data import SliceDataset, assign_labels, describe, scan, split_items, subsample

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description="Alzheimer MRI classification with slice- vs. subject-level splits")
    p.add_argument("--root", required=True, help="Data folder containing the class folders")
    p.add_argument("--split", choices=["slice", "subject"], default="subject")
    p.add_argument("--classes", choices=["binary", "four"], default="binary")
    p.add_argument("--subject-regex", default=r"(OAS1_\d{4})", help="Regex that extracts the subject ID from the file name")
    p.add_argument("--slices-per-subject", type=int, default=20)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="Default: runs/<split>")
    return p.parse_args()


def build_model(n_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probs.append(torch.softmax(model(x.to(device)).float(), dim=1).cpu())
    return torch.cat(probs).numpy()


def metrics(y, prob, n_classes):
    pred = prob.argmax(1)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(n_classes))).tolist(),
    }
    try:
        out["auc"] = roc_auc_score(y, prob[:, 1]) if n_classes == 2 else roc_auc_score(y, prob, multi_class="ovr")
    except ValueError:
        out["auc"] = None
    return out


def subject_level(items, prob):
    groups = defaultdict(list)
    for it, p in zip(items, prob):
        groups[it["subject"]].append((it["y"], p))
    ys, ps = [], []
    for v in groups.values():
        ys.append(v[0][0])
        ps.append(np.mean([p for _, p in v], axis=0))
    return np.array(ys), np.array(ps)


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = a.out or os.path.join("runs", a.split)
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items, classes = assign_labels(scan(a.root, a.subject_regex), a.classes)
    items = subsample(items, a.slices_per_subject)
    train, val, test = split_items(items, a.split, a.seed)
    print(f"Split: {a.split} | Classes: {classes}")
    overlap = describe(train, val, test, classes)

    tf_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(7), transforms.ColorJitter(0.1, 0.1),
                                   transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tf_eval = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dl = lambda part, tf, sh: DataLoader(SliceDataset(part, tf), batch_size=a.batch, shuffle=sh,  # noqa: E731
                                         num_workers=a.workers, pin_memory=device.type == "cuda")
    train_dl, val_dl, test_dl = dl(train, tf_train, True), dl(val, tf_eval, False), dl(test, tf_eval, False)

    n = len(classes)
    counts = np.bincount([it["y"] for it in train], minlength=n).astype(float)
    weights = torch.tensor(counts.sum() / (n * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model = build_model(n).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)  # class-weighted loss for class imbalance
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_f1, history, t0 = -1.0, [], time.time()
    y_val = np.array([it["y"] for it in val])
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
        val_f1 = metrics(y_val, predict(model, val_dl, device), n)["macro_f1"]
        history.append({"epoch": ep, "train_loss": total / len(train), "val_macro_f1": val_f1})
        print(f"Epoch {ep}/{a.epochs} | train loss {total / len(train):.4f} | val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:  # model selection uses the validation set, never the test set
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))

    model.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location=device))
    prob = predict(model, test_dl, device)
    y_test = np.array([it["y"] for it in test])
    ys, ps = subject_level(test, prob)
    result = {
        "split": a.split,
        "classes": classes,
        "train_test_subject_overlap": overlap,
        "test_slices": len(test),
        "test_subjects": len(ys),
        "slice_level": metrics(y_test, prob, n),
        "subject_level": metrics(ys, ps, n),
        "history": history,
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out, "split.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "subject", "y", "split"])
        w.writeheader()
        for it in train + val + test:
            w.writerow({k: it[k] for k in ("path", "subject", "y", "split")})

    s, h = result["slice_level"], result["subject_level"]
    print(f"\n=== Test ({a.split} split) ===")
    print(f"Slice level   : accuracy {s['accuracy']:.3f} | macro-F1 {s['macro_f1']:.3f} | AUC {s['auc']}")
    print(f"Subject level : accuracy {h['accuracy']:.3f} | macro-F1 {h['macro_f1']:.3f} | AUC {h['auc']}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
