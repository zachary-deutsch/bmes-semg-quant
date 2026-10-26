import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import warnings

from bmes_semg.data import Padder, Trial, load_csv, synthetic_trials
from bmes_semg.evaluation import Config, balanced_split, evaluate
from bmes_semg.features import QuantFeatures
from bmes_semg.cli import main

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def write_csv(path, blocks):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "subject", "class", "trial", *[f"emg{i}" for i in range(1, 13)]])
        for subject, label, trial, values in blocks:
            for t, value in enumerate(values):
                writer.writerow([t / 2000, subject, label, trial, *([value] * 12)])


def test_contiguous_repetitions_stay_separate(tmp_path):
    path = tmp_path / "trials.csv"
    write_csv(path, [(1, 1, 1, [1] * 8), (1, 1, 2, [2] * 9)])
    trials = load_csv(path)
    assert len(trials) == 2
    assert trials[0].signal.shape == (12, 8)
    assert trials[1].signal.shape == (12, 9)
    assert trials[1].repetition == 2


@pytest.mark.parametrize("blocks", [
    [(1, 1, 1, [1] * 8), (1, 2, 1, [2] * 8), (1, 1, 1, [3] * 8)],
    [(1, 1, 1, [float('nan')] * 8)],
    [(1, 0, 1, [1] * 8)],
    [],
])
def test_invalid_csv_rejected(tmp_path, blocks):
    path = tmp_path / "trials.csv"
    write_csv(path, blocks)
    with pytest.raises(ValueError):
        load_csv(path)


def test_time_reset_inside_trial_rejected(tmp_path):
    path = tmp_path / "trials.csv"
    write_csv(path, [(1, 1, 1, [1] * 8)])
    lines = path.read_text().splitlines()
    lines[-1] = lines[-1].replace("0.0035", "0.0")
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="Time must"):
        load_csv(path)


def test_padding_uses_training_length_without_mutation():
    a, b = np.ones((12, 8)), np.ones((12, 10))
    padder = Padder().fit([a, b])
    result = padder.transform([a])
    assert result.shape == (1, 12, 10)
    assert (result[..., -2:] == 0).all()
    assert a.shape == (12, 8)
    with pytest.raises(ValueError, match="exceeds"):
        padder.transform([np.ones((12, 11))])
    assert padder.length == 10


def test_splits_are_balanced_disjoint_and_seeded():
    labels = np.repeat([1, 2, 3], 6)
    split = balanced_split(labels, np.random.default_rng(11))
    again = balanced_split(labels, np.random.default_rng(11))
    assert all(np.array_equal(a, b) for a, b in zip(split, again))
    assert len(set(np.concatenate(split))) == 18
    for indices, count in zip(split, (4, 1, 1)):
        assert np.array_equal(np.bincount(labels[indices])[1:], [count] * 3)
    with pytest.raises(ValueError, match="exactly 6"):
        balanced_split(labels[:-1], np.random.default_rng(11))


@pytest.mark.parametrize("length,depth", [(8, 1), (8, 6), (64, 4), (128, 2)])
def test_operational_quant_matches_recovered_original(length, depth):
    spec = importlib.util.spec_from_file_location("recovered_quant", ROOT / "archive/2025-06-01_quant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Reviewed: definitions/imports only, no I/O.
    x = np.random.default_rng(1).normal(size=(6, 12, length)).astype(np.float32)
    # The immutable historical code warns on modern NumPy/Torch scalar interop.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        expected = module.Quant(depth=depth, div=4).fit_transform(torch.from_numpy(x), None).numpy()
    assert all("__array_wrap__" in str(w.message) for w in captured)
    wrapper = QuantFeatures(depth=depth, div=4)
    actual = wrapper.fit_transform(x)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(wrapper.transform(x), actual)
    assert np.isfinite(actual).all()


def test_quant_rejects_shape_changes_and_nonfinite():
    quant = QuantFeatures()
    with pytest.raises(ValueError, match="not fitted"):
        quant.transform(np.ones((2, 12, 64)))
    quant.fit_transform(np.ones((2, 12, 64)))
    with pytest.raises(ValueError, match="match training"):
        quant.transform(np.ones((2, 12, 65)))
    with pytest.raises(ValueError, match="finite"):
        quant.transform(np.full((2, 12, 64), np.nan))


def test_evaluation_is_deterministic_and_subject_specific():
    trials = synthetic_trials()
    config = Config(resamples=1, estimators=5)
    a, b = evaluate(trials, config), evaluate(trials, config)
    assert a["summary"] == b["summary"]
    for first, second in zip(a["runs"], b["runs"]):
        assert first["split_keys"] == second["split_keys"]
        keys = [tuple(k) for part in first["split_keys"].values() for k in part]
        assert len(keys) == len(set(keys)) == 18
        assert {k[0] for k in keys} == {first["subject"]}
        assert all(v >= 0 for v in first["timing_seconds"].values())
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(trials + trials[:1], config)


def test_scaling_is_fitted_only_on_training(monkeypatch):
    import bmes_semg.evaluation as evaluation
    actual_scaler = evaluation.StandardScaler
    fitted_rows = []

    class RecordingScaler(actual_scaler):
        def fit(self, x, y=None, **kwargs):
            fitted_rows.append(len(x))
            return super().fit(x, y, **kwargs)

    monkeypatch.setattr(evaluation, "StandardScaler", RecordingScaler)
    evaluate(synthetic_trials(), Config(resamples=1, estimators=2))
    assert fitted_rows == [12, 12]  # 3 classes x 4 training trials per subject.


def test_changing_heldout_values_cannot_change_classifier_training(monkeypatch):
    import bmes_semg.evaluation as evaluation
    actual_trees = evaluation.ExtraTreesClassifier
    fitted_features = []

    class RecordingTrees(actual_trees):
        def fit(self, x, y, **kwargs):
            fitted_features.append(np.asarray(x).copy())
            return super().fit(x, y, **kwargs)

    monkeypatch.setattr(evaluation, "ExtraTreesClassifier", RecordingTrees)
    trials = [t for t in synthetic_trials() if t.subject == 1]
    config = Config(resamples=1, estimators=2)
    _, validation, test = balanced_split(np.asarray([t.label for t in trials]),
                                        np.random.default_rng(config.seed))
    heldout = set(np.concatenate((validation, test)).tolist())
    changed = [Trial(t.subject, t.label, t.repetition,
                     t.signal + 1000 if i in heldout else t.signal.copy())
               for i, t in enumerate(trials)]
    evaluate(trials, config)
    evaluate(changed, config)
    np.testing.assert_array_equal(fitted_features[0], fitted_features[1])


def test_cli_smoke_has_honest_label_and_preserves_existing_output(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"resamples": 1, "estimators": 2}))
    output = tmp_path / "report.json"
    args = ["--synthetic", "--config", str(config), "--output", str(output)]
    main(args)
    report = json.loads(output.read_text())
    assert report["status"] == "SYNTHETIC FUNCTIONALITY ONLY - NOT PAPER RESULTS"
    assert len(report["runs"]) == 4
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == before


def test_csv_loader_and_pipeline_integration(tmp_path):
    path = tmp_path / "trials.csv"
    blocks = [(1, label, rep, [float(label)] * 16)
              for label in (1, 2, 3) for rep in range(1, 7)]
    write_csv(path, blocks)
    report = evaluate(load_csv(path), Config(resamples=1, estimators=2, max_features=4))
    assert len(report["runs"]) == 2
    assert report["runs"][0]["features"] > 0


def test_archive_integrity():
    expected = {
        "2025-06-01_quant.py": "68bb354a6fd2344c5a160a141feb9db4599fce2bad5e4acf5381c435b3b8772a",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((ROOT / "archive" / filename).read_bytes()).hexdigest() == digest
    manifest = json.loads((ROOT / "archive/manifest.json").read_text())
    for item in manifest["sources"]:
        assert hashlib.sha256((ROOT / "archive" / item["file"]).read_bytes()).hexdigest() == item["sha256"]
