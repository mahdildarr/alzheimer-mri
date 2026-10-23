import numpy as np
import pytest

from admri.data import assign_labels, cv_folds, permute_subject_labels, scan, split_items, subsample
from admri.metrics import bootstrap_ci, classification_metrics, expected_calibration_error, subject_level


def make_items(n_subjects=30, slices=6):
    items = []
    for s in range(n_subjects):
        for k in range(slices):
            items.append({"path": f"s{s}_{k}.jpg", "folder": "x", "subject": f"OAS1_{s:04d}", "y": s % 2})
    return items


def test_scan_and_labels(tmp_path):
    for folder, sid in (("Non Demented", "0001"), ("Mild Dementia", "0002"), ("Mild Dementia", "0001")):
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / f"OAS1_{sid}_MR1_mpr-1_100.jpg").write_bytes(b"")
    (tmp_path / "Mild Dementia" / "no_id.jpg").write_bytes(b"")
    items, classes = assign_labels(scan(tmp_path, r"(OAS1_\d{4})"), "binary")
    assert classes == ["No dementia", "Dementia"]
    # subject 0001 appears under two classes and is removed
    assert [(it["subject"], it["y"]) for it in items] == [("OAS1_0002", 1)]


def test_subsample_is_evenly_spaced():
    items = [{"path": f"OAS1_0001_{k}.jpg", "subject": "a", "y": 0} for k in range(100)]
    picked = [int(it["path"].split("_")[-1][:-4]) for it in subsample(items, 5)]
    assert picked == [0, 25, 50, 74, 99]


@pytest.mark.parametrize("mode,expect_overlap", [("subject", False), ("slice", True)])
def test_split_protocols(mode, expect_overlap):
    tr, va, te = split_items(make_items(), mode, seed=0)
    overlap = {it["subject"] for it in tr} & {it["subject"] for it in te}
    assert bool(overlap) == expect_overlap
    assert len(tr) + len(va) + len(te) == 180


def test_subject_cv_tests_every_subject_once():
    items = make_items(40)
    tested = []
    for tr, va, te in cv_folds(items, 5, "subject", seed=0):
        s_tr, s_va, s_te = ({it["subject"] for it in part} for part in (tr, va, te))
        assert not (s_tr & s_te) and not (s_va & s_te) and not (s_tr & s_va)
        assert {it["y"] for it in te} == {0, 1}  # stratified
        tested += sorted(s_te)
    assert sorted(tested) == sorted({it["subject"] for it in items})


def test_permutation_keeps_subjects_consistent():
    items = make_items(40)
    perm = permute_subject_labels(items, seed=1)
    by_subject = {}
    for it in perm:
        by_subject.setdefault(it["subject"], set()).add(it["y"])
    assert all(len(v) == 1 for v in by_subject.values())
    assert sum(it["y"] for it in perm) == sum(it["y"] for it in items)  # class balance is unchanged
    assert [it["y"] for it in perm] != [it["y"] for it in items]


def test_metrics_and_calibration():
    y = np.array([0, 0, 1, 1])
    perfect = np.eye(2)[y]
    m = classification_metrics(y, perfect, 2)
    assert m["accuracy"] == 1 and m["auc"] == 1 and m["brier"] == 0 and m["ece"] == 0
    overconfident = np.array([[0.99, 0.01], [0.01, 0.99], [0.01, 0.99], [0.99, 0.01]])
    assert expected_calibration_error(y, overconfident) == pytest.approx(0.49, abs=1e-6)


def test_subject_level_averages_slices():
    items = [{"subject": "a", "y": 1}, {"subject": "a", "y": 1}, {"subject": "b", "y": 0}]
    subjects, ys, ps = subject_level(items, np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]]))
    assert subjects == ["a", "b"] and ys.tolist() == [1, 0]
    np.testing.assert_allclose(ps[0], [0.4, 0.6])


def test_bootstrap_interval_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p1 = np.clip(y * 0.6 + rng.uniform(0, 0.4, 200), 0, 1)
    prob = np.c_[1 - p1, p1]
    lo, hi = bootstrap_ci(y, prob, 2, "balanced_accuracy", n=200)
    est = classification_metrics(y, prob, 2)["balanced_accuracy"]
    assert lo <= est <= hi
