"""Fixed-config subject-specific evaluation; no model selection on held-out test."""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from .data import Padder
from .features import QuantFeatures


@dataclass(frozen=True)
class Config:
    seed: int = 42
    resamples: int = 2
    depth: int = 2
    div: int = 4
    estimators: int = 200
    max_features: int = 32
    pad_length: int | None = None

    def __post_init__(self):
        for name in ("resamples", "estimators", "max_features", "depth", "div"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        QuantFeatures(self.depth, self.div)
        Padder(self.pad_length)


def balanced_split(labels, rng):
    """Exactly 4/1/1 per class; reject missing or surplus repetitions."""
    parts = [[], [], []]
    if len(np.unique(labels)) < 2:
        raise ValueError("At least two classes are required")
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) != 6:
            raise ValueError(f"Class {label} has {len(indices)} trials; require exactly 6")
        indices = rng.permutation(indices)
        for part, values in zip(parts, (indices[:4], indices[4:5], indices[5:6])):
            part.extend(values.tolist())
    return tuple(np.asarray(part, dtype=int) for part in parts)


def evaluate(trials, config: Config):
    """Fit one ExtraTrees and one illustrative Ridge model per subject/resample.

    Ridge alpha=1 is a new demo choice; the poster-final alpha is unknown.
    Scaling and padding are fitted on training trials, never on held-out values.
    """
    if not trials:
        raise ValueError("No trials provided")
    keys = [(t.subject, t.label, t.repetition) for t in trials]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate subject/class/trial keys")
    rng = np.random.default_rng(config.seed)
    rows = []
    for subject in sorted({t.subject for t in trials}):
        selected = [t for t in trials if t.subject == subject]
        labels = np.asarray([t.label for t in selected])
        for resample in range(config.resamples):
            split = balanced_split(labels, rng)
            batches = [[selected[i].signal for i in indices] for indices in split]
            t0 = perf_counter()
            padder = Padder(config.pad_length).fit(batches[0])
            quant = QuantFeatures(config.depth, config.div)
            z_train = quant.fit_transform(padder.transform(batches[0]))
            scaler = StandardScaler().fit(z_train)
            z_train = scaler.transform(z_train)
            train_time = perf_counter() - t0
            heldout, heldout_time = [], []
            for batch in batches[1:]:
                start = perf_counter()
                heldout.append(scaler.transform(quant.transform(padder.transform(batch))))
                heldout_time.append(perf_counter() - start)
            if config.max_features > z_train.shape[1]:
                raise ValueError("max_features exceeds the QUANT feature count")
            models = {
                "ExtraTrees": ExtraTreesClassifier(
                    n_estimators=config.estimators, max_features=config.max_features,
                    criterion="entropy", random_state=config.seed + resample, n_jobs=1,
                ),
                "Ridge_demo_alpha1": RidgeClassifier(alpha=1.0),
            }
            for name, model in models.items():
                start = perf_counter()
                model.fit(z_train, labels[split[0]])
                fit_time = perf_counter() - start
                start = perf_counter()
                val_predictions = model.predict(heldout[0])
                test_predictions = model.predict(heldout[1])
                row = {
                    "subject": subject, "resample": resample, "model": name,
                    "features": z_train.shape[1], "padded_length": padder.length,
                    "validation_accuracy": float(np.mean(val_predictions == labels[split[1]])),
                    "test_accuracy": float(np.mean(test_predictions == labels[split[2]])),
                    "split_keys": {
                        part: [[selected[i].subject, selected[i].label, selected[i].repetition] for i in idx]
                        for part, idx in zip(("train", "validation", "test"), split)
                    },
                    "timing_seconds": {
                        "train_transform_shared": train_time,
                        "validation_transform_shared": heldout_time[0],
                        "test_transform_shared": heldout_time[1],
                        "classifier_fit": fit_time,
                        "prediction": perf_counter() - start,
                    },
                }
                rows.append(row)
    summary = {}
    for model in models:
        selected_rows = [r for r in rows if r["model"] == model]
        # Equal weighting across subjects; std of subject means, ddof=0.
        summary[model] = {}
        for metric in ("validation_accuracy", "test_accuracy"):
            means = [np.mean([r[metric] for r in selected_rows if r["subject"] == s])
                     for s in sorted({r["subject"] for r in selected_rows})]
            summary[model][metric] = {"mean": float(np.mean(means)), "subject_std_ddof0": float(np.std(means))}
    return {"config": asdict(config), "summary": summary, "runs": rows}
