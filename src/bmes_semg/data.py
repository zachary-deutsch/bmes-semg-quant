"""Explicit trial boundaries and training-only padding; no raw DB7 preprocessing."""
from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trial:
    subject: int
    label: int
    repetition: int
    signal: np.ndarray  # channels x samples


def load_csv(path: Path) -> list[Trial]:
    """Read movement-only, contiguous (subject, class, trial) blocks of 12 channels.

    A repeated key after another block is rejected rather than silently treated
    as an independent trial. Time must increase within each block.
    """
    channels = [f"emg{i}" for i in range(1, 13)]
    required = {"time", "subject", "class", "trial", *channels}
    trials, seen, samples = [], set(), []
    key, last_time = None, None

    def finish():
        if key is not None:
            trials.append(Trial(*key, np.asarray(samples, dtype=np.float32).T.copy()))
            seen.add(key)

    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"CSV needs unique columns: {sorted(required)}")
        for line, row in enumerate(reader, 2):
            try:
                new_key = tuple(int(row[c]) for c in ("subject", "class", "trial"))
                time = float(row["time"])
                sample = [float(row[c]) for c in channels]
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("inconsistent column count")
                if min(new_key) < 1 or not np.isfinite([time, *sample]).all():
                    raise ValueError("IDs must be positive; time/signals must be finite")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid CSV row {line}: {exc}") from exc
            if new_key != key:
                finish()
                if new_key in seen:
                    raise ValueError(f"Noncontiguous duplicate trial key: {new_key}")
                key, samples, last_time = new_key, [], None
            if last_time is not None and time <= last_time:
                raise ValueError(f"Time must strictly increase within trial {key}, row {line}")
            samples.append(sample)
            last_time = time
        finish()
    if not trials:
        raise ValueError("CSV contains no trials")
    return trials


class Padder:
    """Right-pad to a declared length, or a length fitted on training trials only.

    Longer held-out inputs raise an error; they are never silently truncated.
    """
    def __init__(self, length: int | None = None):
        if length is not None and (type(length) is not int or length < 8):
            raise ValueError("Padding length must be an integer >= 8")
        self.declared_length = length

    def fit(self, signals):
        self._validate(signals)
        self.channels = signals[0].shape[0]
        self.length = self.declared_length or max(x.shape[1] for x in signals)
        if self.length < 8:
            raise ValueError("QUANT wrapper requires at least 8 time samples")
        self.transform(signals)
        return self

    @staticmethod
    def _validate(signals):
        if not signals:
            raise ValueError("No trials provided")
        for x in signals:
            if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
                raise ValueError("Trials must be finite, nonempty channels x samples arrays")

    def transform(self, signals):
        if not hasattr(self, "length"):
            raise ValueError("Padder is not fitted")
        self._validate(signals)
        padded = []
        for x in signals:
            if x.shape[0] != self.channels:
                raise ValueError("Channel count changed")
            if x.shape[1] > self.length:
                raise ValueError("Trial exceeds training/declared padding length; resolve preprocessing explicitly")
            padded.append(np.pad(x, ((0, 0), (0, self.length - x.shape[1]))))
        return np.asarray(padded, dtype=np.float32)


def synthetic_trials(seed: int = 42) -> list[Trial]:
    """Toy amplitude/frequency classes; no physiological or benchmark validity."""
    rng = np.random.default_rng(seed)
    t = np.arange(128, dtype=np.float32) / 128
    trials = []
    for subject in (1, 2):
        for label in (1, 2, 3):
            for repetition in range(1, 7):
                x = np.stack([
                    (0.3 * label + c / 50) * np.sin(2 * np.pi * (label + c / 10) * t)
                    + rng.normal(0, 0.08, len(t))
                    for c in range(12)
                ]).astype(np.float32)
                trials.append(Trial(subject, label, repetition, x))
    return trials
