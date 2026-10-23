# Subject-Level Evaluation for Alzheimer's MRI Classification

[![CI](https://github.com/mahdildarr/alzheimer-mri/actions/workflows/ci.yml/badge.svg)](https://github.com/mahdildarr/alzheimer-mri/actions/workflows/ci.yml)

**Measuring the effect of data leakage** in dementia classification from brain MRI slices: the same model is trained with slice-level and subject-level data splits and compared, and the regions it focuses on are visualized with **Grad-CAM**.

<p align="center"><img src="docs/assets/pipeline.svg" width="100%" alt="Pipeline"></p>

> This is an independent learning and reproduction project on publicly available data. It is not intended for clinical use.

## Problem

Popular Alzheimer's MRI datasets on Kaggle consist of many 2D slices extracted from each subject's brain volume. When slices are split at random, nearly identical neighboring slices of the same subject end up in both the training and test sets. The model learns to recognize the *subject* rather than the *disease*, and unrealistically high accuracies (often above 95%) are reported. A sound evaluation requires a **subject-level split**, where test subjects are never seen during training.

## Approach

| Step | Method | Module |
|---|---|---|
| 1. Data preparation | Extract the subject ID from the file name; select N evenly spaced slices per subject; binary (no dementia / dementia) or four-class labels | `admri.data` |
| 2. Slice-level split | Slices are split at random (leaky, common practice) | `admri.train --split slice` |
| 3. Subject-level split | Subjects are split; all slices of a subject stay in one set | `admri.train --split subject` |
| 4. Model | ImageNet-pretrained ResNet-18, class-weighted loss; the epoch is selected on *validation subjects* (subject-level balanced accuracy), never on the test set | `admri.train` |
| 5. Subject-level decision | One decision per subject from the mean of its slice probabilities, with a 95% bootstrap interval over subjects | `admri.metrics` |
| 6. Cross-validation | Stratified group K-fold over subjects: every subject is tested exactly once by a model that never saw it; both protocols run on the same folds and seeds, so the leakage gap is a paired difference | `admri.cv` |
| 7. Leakage probe | Labels are shuffled *between subjects*: there is then nothing to learn about dementia, and any test accuracy above chance can only come from recognising subjects seen during training | `--permute-labels`, `admri.cv --permute` |
| 8. Calibration | Brier score and expected calibration error (ECE) of the subject-level probabilities | `admri.metrics` |
| 9. Interpretability | Grad-CAM heat maps, plus the share of heat-map energy that falls inside the brain (a model that attends to the skull outline or image borders scores low) | `admri.gradcam` |

## Data

- [OASIS Alzheimer's Detection](https://www.kaggle.com/datasets/ninadaithal/imagesoasis) (Kaggle): 2D slices derived from the OASIS-1 cross-sectional MRI data; classes based on CDR: Non Demented, Very mild Dementia, Mild Dementia, Moderate Dementia.
- The subject ID is read from the `OAS1_XXXX` pattern in the file name (configurable via `--subject-regex`).
- Because the moderate dementia class contains very few subjects, the default evaluation is **binary**: no dementia (CDR 0) vs. dementia (CDR ≥ 0.5).
- No data is included in this repository; the data is subject to the OASIS data use terms.

## Installation

```bash
git clone https://github.com/mahdildarr/alzheimer-mri.git
cd alzheimer-mri
pip install -e ".[dev]"
pytest -q          # unit tests + end-to-end runs on synthetic slices (CPU, ~2 min)
```

The quickest way to run everything is the `notebooks/colab_alzheimer_mri.ipynb` notebook on Google Colab (GPU enabled).

## Usage

```bash
# 1) Same settings, two different splits (one hold-out split each)
python -m admri.train --root <data folder> --split slice --epochs 8
python -m admri.train --root <data folder> --split subject --epochs 8

# 2) Leakage probe: labels permuted between subjects
python -m admri.train --root <data folder> --split slice --permute-labels --epochs 8
python -m admri.train --root <data folder> --split subject --permute-labels --epochs 8
python -m admri.report --runs runs/slice runs/subject runs/slice_permuted runs/subject_permuted

# 3) 5-fold cross-validation of both protocols, with and without permuted labels
python -m admri.cv --root <data folder> --folds 5 --epochs 8 --permute --out runs/cv

# 4) Grad-CAM, with the in-brain energy share over all test slices
python -m admri.gradcam --run runs/subject --n 8 --all
```

No data yet? `python tools/make_synthetic_oasis.py --out data/synthetic --signal 0` creates slices in the same
layout, where each subject has its own "anatomy" and there is no disease signal at all (`--signal 0`).

## Evaluation metrics

- Slice level and subject level: accuracy, balanced accuracy, macro-F1, ROC-AUC, confusion matrix
- Subject level: 95% bootstrap interval of the balanced accuracy (subjects are resampled), Brier score, ECE
- Cross-validation: mean ± std over folds, and a pooled estimate over all out-of-fold subject predictions
- Leakage check: number of subjects present in both train and test sets (must be 0 for the subject-level split), and the permuted-label probe (must be at chance for the subject-level split)

## Sanity check on synthetic data

On synthetic slices with **no disease signal** (30 subjects × 10 slices, `--signal 0`, small CNN, 8 epochs on a CPU), the
only thing a model can learn is which subject a slice comes from:

| Split | Train ∩ test subjects | Subject-level balanced accuracy (95% CI) |
|---|---|---|
| Slice-level (leaky) | > 0 | 0.79 (0.65–0.94) |
| Subject-level | 0 | 0.50 (chance) |

The leaky split reports a convincing result for a task with nothing to learn. This is the effect the OASIS experiments
measure on real data; the same check runs in the test suite on every commit.

## Progress

- [x] Subject ID parsing and slice selection
- [x] Slice-level and subject-level splits
- [x] ResNet-18 transfer learning with class weighting
- [x] Subject-level decisions from slice probabilities
- [x] Grad-CAM visualisation, with the share of attention inside the brain
- [x] Subject-level stratified K-fold cross-validation with paired slice vs subject comparison
- [x] Permuted-label leakage probe
- [x] Bootstrap confidence intervals and calibration (Brier, ECE)
- [x] Unit tests, end-to-end tests on synthetic data and CI
- [x] Colab notebook

Next:

- [ ] Binary classification experiments on OASIS (cross-validated)
- [ ] Effect of the number of slices per subject

## Project structure

```
├── notebooks/colab_alzheimer_mri.ipynb
├── src/admri/
│   ├── data.py       # scanning, subject IDs, slice selection, splits, CV folds, label permutation
│   ├── models.py     # ResNet-18 (pretrained / scratch), small CNN
│   ├── metrics.py    # slice / subject metrics, calibration, bootstrap intervals
│   ├── train.py      # one training run with slice- or subject-level split
│   ├── cv.py         # cross-validated comparison of both protocols
│   ├── report.py     # comparison table of single runs
│   └── gradcam.py    # Grad-CAM and in-brain energy share
├── tests/            # unit tests and end-to-end runs on synthetic data
├── tools/make_synthetic_oasis.py
└── pyproject.toml
```

## References

- Marcus et al., *Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults*, Journal of Cognitive Neuroscience, 2007
- Wen et al., *Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation*, Medical Image Analysis, 2020
- [3D MRI-Based Alzheimer's Disease Classification Using Multi-Modal 3D CNN with Leakage-Aware Subject-Level Evaluation](https://arxiv.org/abs/2603.17304), arXiv, 2026
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017

## License

MIT (code). The data is subject to the OASIS data use terms.
