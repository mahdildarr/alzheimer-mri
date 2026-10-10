"""Dataset scanning, subject ID extraction, slice selection and train / validation / test splits."""

import os
import re
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

IMG_EXT = (".jpg", ".jpeg", ".png")


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def scan(root, subject_regex):
    """Scan images; the class comes from the folder name and the subject ID from the file name."""
    pat = re.compile(subject_regex)
    items, missing = [], 0
    for d, _, files in os.walk(root):
        for f in sorted(files, key=natural_key):
            if not f.lower().endswith(IMG_EXT):
                continue
            m = pat.search(f)
            if not m:
                missing += 1
                continue
            items.append({"path": os.path.join(d, f), "folder": os.path.basename(d), "subject": m.group(1)})
    if not items:
        raise SystemExit(f"No images with a subject ID found. Check the file names and --subject-regex: {root}")
    if missing:
        print(f"Warning: skipped {missing} images without a readable subject ID.")
    return items


def label_map(folders, mode):
    folders = sorted(set(folders), key=natural_key)
    if mode == "binary":
        return {f: (0 if "non" in f.lower() else 1) for f in folders}, ["No dementia", "Dementia"]
    return {f: i for i, f in enumerate(folders)}, folders


def assign_labels(items, mode):
    mapping, classes = label_map([it["folder"] for it in items], mode)
    by_subject = defaultdict(list)
    for it in items:
        it["y"] = mapping[it["folder"]]
        by_subject[it["subject"]].append(it["y"])
    bad = {s for s, ys in by_subject.items() if len(set(ys)) > 1}
    if bad:
        print(f"Warning: removed {len(bad)} subjects that appear under more than one class.")
        items = [it for it in items if it["subject"] not in bad]
    return items, classes


def subsample(items, per_subject):
    """Pick at most N evenly spaced slices per subject (balances the number of images per subject)."""
    if not per_subject:
        return items
    groups = defaultdict(list)
    for it in items:
        groups[it["subject"]].append(it)
    out = []
    for g in groups.values():
        g = sorted(g, key=lambda x: natural_key(os.path.basename(x["path"])))
        idx = np.unique(np.linspace(0, len(g) - 1, min(per_subject, len(g))).round().astype(int))
        out += [g[i] for i in idx]
    return out


def _split(units, labels, size, seed):
    try:
        return train_test_split(units, test_size=size, stratify=labels, random_state=seed)
    except ValueError:  # stratification is impossible for classes with very few members
        return train_test_split(units, test_size=size, random_state=seed)


def split_items(items, mode, seed=42, val=0.15, test=0.15):
    """mode='slice': slices are split at random (slices of one subject can land in different sets → leakage).
    mode='subject': subjects are split; all slices of a subject stay in a single set."""
    if mode == "slice":
        idx = list(range(len(items)))
        y = [it["y"] for it in items]
        tr, rest = _split(idx, y, val + test, seed)
        va, te = _split(rest, [y[i] for i in rest], test / (val + test), seed)
    else:
        sy = {it["subject"]: it["y"] for it in items}
        subjects = sorted(sy)
        tr_s, rest_s = _split(subjects, [sy[s] for s in subjects], val + test, seed)
        va_s, te_s = _split(rest_s, [sy[s] for s in rest_s], test / (val + test), seed)
        part = {**{s: "train" for s in tr_s}, **{s: "val" for s in va_s}, **{s: "test" for s in te_s}}
        tr = [i for i, it in enumerate(items) if part[it["subject"]] == "train"]
        va = [i for i, it in enumerate(items) if part[it["subject"]] == "val"]
        te = [i for i, it in enumerate(items) if part[it["subject"]] == "test"]
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        for i in ids:
            items[i]["split"] = name
    return [items[i] for i in tr], [items[i] for i in va], [items[i] for i in te]


def describe(train, val, test, classes):
    s_tr, s_te = {it["subject"] for it in train}, {it["subject"] for it in test}
    for name, part in (("Train", train), ("Validation", val), ("Test", test)):
        c = Counter(it["y"] for it in part)
        print(f"{name:<10}: {len(part):>6} slices, {len({it['subject'] for it in part}):>4} subjects | "
              + ", ".join(f"{classes[k]}: {c.get(k, 0)}" for k in range(len(classes))))
    overlap = len(s_tr & s_te)
    print(f"Subjects present in both train and test: {overlap}")
    return overlap


class SliceDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return self.transform(Image.open(it["path"]).convert("RGB")), it["y"]
