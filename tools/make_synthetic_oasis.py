"""Create a small synthetic dataset in the layout of the Kaggle OASIS slices to exercise the pipeline.

Every subject gets an individual "anatomy" (head shape, ventricle position, texture) that all its
slices share, and demented subjects get enlarged ventricles scaled by ``--signal``. With
``--signal 0`` there is nothing to learn about the disease, only about the subjects, which makes
subject leakage directly visible.

    python tools/make_synthetic_oasis.py --out data/synthetic --subjects 40 --slices 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["Non Demented", "Very mild Dementia"]


def subject_slices(rng, demented: bool, n: int, size: int, signal: float):
    yy, xx = np.mgrid[0:size, 0:size] / size - 0.5
    a, b = rng.uniform(0.30, 0.42, 2)                 # head shape
    vx, vy = rng.uniform(-0.08, 0.08, 2)              # ventricle position
    vr = 0.05 + (signal * 0.06 if demented else 0.0)  # enlarged ventricles with dementia
    # subject "fingerprint": a smooth intensity pattern shared by all slices, like individual anatomy
    f1, f2 = rng.uniform(2, 6, 2) * 2 * np.pi
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    texture = np.sin(f1 * xx + p1) * np.cos(f2 * yy + p2)
    for k in range(n):
        z = (k - n / 2) / n                           # position along the axial axis
        head = (xx / a) ** 2 + (yy / b) ** 2 < 1 - z**2
        vent = ((xx - vx) ** 2 + (yy - vy) ** 2) < (vr * (1 - abs(z))) ** 2
        img = head * (0.55 + 0.35 * texture) - vent * 0.5 + rng.normal(0, 0.03, (size, size))
        yield (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--subjects", type=int, default=40)
    p.add_argument("--slices", type=int, default=12)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--signal", type=float, default=1.0, help="Strength of the disease effect (0 = none)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    for s in range(1, a.subjects + 1):
        demented = s % 2 == 0
        folder = Path(a.out) / CLASSES[int(demented)]
        folder.mkdir(parents=True, exist_ok=True)
        for k, img in enumerate(subject_slices(rng, demented, a.slices, a.size, a.signal)):
            Image.fromarray(img).save(folder / f"OAS1_{s:04d}_MR1_mpr-1_{100 + k}.jpg")
    print(f"{a.subjects} subjects x {a.slices} slices -> {a.out}")


if __name__ == "__main__":
    main()
