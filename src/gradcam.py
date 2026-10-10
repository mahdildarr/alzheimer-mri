"""Grad-CAM visualization of the regions the model focuses on when making a decision.

An equal number of slices per class is taken from the test set of the subject-level split.

Example:
    python src/gradcam.py --run runs/subject --n 8
"""

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

from train import MEAN, STD, build_model  # noqa: E402


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


def main():
    p = argparse.ArgumentParser(description="Grad-CAM visualization")
    p.add_argument("--run", default="runs/subject")
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()

    meta = json.load(open(os.path.join(a.run, "metrics.json"), encoding="utf-8"))
    classes = meta["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes))
    model.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=device))
    model.to(device).eval()

    rows = [r for r in csv.DictReader(open(os.path.join(a.run, "split.csv"), encoding="utf-8")) if r["split"] == "test"]
    per_class = max(1, a.n // len(classes))
    chosen = []
    for c in range(len(classes)):
        rc = [r for r in rows if int(r["y"]) == c]
        if rc:
            chosen += [rc[i] for i in np.linspace(0, len(rc) - 1, min(per_class, len(rc))).astype(int)]

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    fig, axes = plt.subplots(2, len(chosen), figsize=(2.4 * len(chosen), 5.2))
    axes = np.array(axes).reshape(2, -1)
    for j, r in enumerate(chosen):
        img = Image.open(r["path"]).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        cam, pred, conf = grad_cam(model, x, model.layer4)
        base = np.array(img.resize((224, 224))) / 255.0
        axes[0, j].imshow(base)
        axes[0, j].set_title(f"True: {classes[int(r['y'])]}", fontsize=8)
        axes[1, j].imshow(base)
        axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, j].set_title(f"Pred: {classes[pred]} ({conf:.2f})", fontsize=8)
        for ax in axes[:, j]:
            ax.axis("off")
    plt.tight_layout()
    out = os.path.join(a.run, "gradcam.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
