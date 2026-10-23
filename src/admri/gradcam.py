"""Grad-CAM visualization of the regions the model focuses on when making a decision.

An equal number of slices per class is taken from the test set of the subject-level split.

Besides the heat maps, the share of Grad-CAM energy that falls inside the brain (non-background
pixels) is reported: a model that relies on the skull outline, image borders or scanner artefacts
puts much of its attention outside the brain.

Example::

    python -m admri.gradcam --run runs/subject --n 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from .models import MEAN, STD, build_model, gradcam_layer  # noqa: E402


def grad_cam(model, x, target_layer, cls=None):
    feats, grads = {}, {}

    def hook(module, inp, out):
        feats["v"] = out
        out.register_hook(lambda g: grads.__setitem__("v", g))

    handle = target_layer.register_forward_hook(hook)
    logits = model(x)
    cls = int(logits.argmax(1)) if cls is None else cls
    model.zero_grad()
    logits[0, cls].backward()
    handle.remove()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats["v"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), cls, torch.softmax(logits, 1)[0, cls].item()


def brain_mask(img: np.ndarray, thr: float = 0.08) -> np.ndarray:
    """Pixels brighter than the (near-black) background of a skull-stripped slice."""
    g = img.mean(axis=2) if img.ndim == 3 else img
    return g > thr * g.max()


def energy_inside(cam: np.ndarray, mask: np.ndarray) -> float:
    return float(cam[mask].sum() / max(cam.sum(), 1e-8))


def main(argv=None):
    p = argparse.ArgumentParser(description="Grad-CAM visualization")
    p.add_argument("--run", default="runs/subject")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--all", action="store_true", help="Compute the in-brain energy share over all test slices")
    a = p.parse_args(argv)

    meta = json.load(open(os.path.join(a.run, "metrics.json"), encoding="utf-8"))
    classes = meta["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = meta.get("args", {})
    model = build_model(args.get("model", "resnet18"), len(classes))
    size = args.get("img_size", 224)
    model.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=device))
    model.to(device).eval()

    rows = [r for r in csv.DictReader(open(os.path.join(a.run, "split.csv"), encoding="utf-8")) if r["split"] == "test"]
    per_class = max(1, a.n // len(classes))
    chosen = []
    for c in range(len(classes)):
        rc = [r for r in rows if int(r["y"]) == c]
        if rc:
            chosen += [rc[i] for i in np.linspace(0, len(rc) - 1, min(per_class, len(rc))).astype(int)]

    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    layer = gradcam_layer(model)
    fig, axes = plt.subplots(2, len(chosen), figsize=(2.4 * len(chosen), 5.2))
    axes = np.array(axes).reshape(2, -1)
    for j, r in enumerate(chosen):
        img = Image.open(r["path"]).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        cam, pred, conf = grad_cam(model, x, layer)
        base = np.array(img.resize((size, size))) / 255.0
        axes[0, j].imshow(base)
        axes[0, j].set_title(f"True: {classes[int(r['y'])]}", fontsize=8)
        axes[1, j].imshow(base)
        axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
        share = energy_inside(cam, brain_mask(base))
        axes[1, j].set_title(f"Pred: {classes[pred]} ({conf:.2f})\nin brain {share:.0%}", fontsize=8)
        for ax in axes[:, j]:
            ax.axis("off")
    plt.tight_layout()
    out = os.path.join(a.run, "gradcam.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if a.all:
        shares = []
        for r in rows:
            img = Image.open(r["path"]).convert("RGB")
            cam, _, _ = grad_cam(model, tf(img).unsqueeze(0).to(device), layer)
            shares.append(energy_inside(cam, brain_mask(np.array(img.resize((size, size))) / 255.0)))
        stats = {"slices": len(shares), "mean_energy_in_brain": float(np.mean(shares)),
                 "share_of_slices_below_50pct": float(np.mean(np.array(shares) < 0.5))}
        json.dump(stats, open(os.path.join(a.run, "gradcam_stats.json"), "w"), indent=2)
        print(stats)


if __name__ == "__main__":
    main()
