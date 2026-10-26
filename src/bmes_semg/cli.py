"""Local-only CLI. Never downloads, uploads, or persists a trained model."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import time_ns

import torch

from .data import load_csv, synthetic_trials
from .evaluation import Config, evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="Toy functionality demo; not BMES results")
    source.add_argument("--csv", type=Path, help="Locally prepared movement-only all_trials.csv")
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.json"))
    parser.add_argument("--output", type=Path, help="New JSON path; default: unique file under outputs/")
    args = parser.parse_args(argv)
    if args.output is None:
        kind = "synthetic" if args.synthetic else "local_csv"
        args.output = Path("outputs") / f"{kind}_{time_ns()}.json"
    if not args.synthetic and args.output.name.startswith("synthetic"):
        parser.error("--csv requires a distinct --output path to keep synthetic evidence labeled correctly")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new path or move it first")
    try:
        config = Config(**json.loads(args.config.read_text()))
        torch.set_num_threads(1)
        trials = synthetic_trials(config.seed) if args.synthetic else load_csv(args.csv)
        report = evaluate(trials, config)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    input_digest = None
    if args.csv is not None:
        with args.csv.open("rb") as stream:
            input_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report.update({
        "status": "SYNTHETIC FUNCTIONALITY ONLY - NOT PAPER RESULTS" if args.synthetic
                  else "LOCAL REAL-DATA CHECK - NOT VERIFIED POSTER REPRODUCTION",
        "data_source": "deterministic synthetic generator" if args.synthetic else "user-supplied local CSV",
        "input_sha256": input_digest,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{p: version(p) for p in ("numpy", "scipy", "scikit-learn", "torch")}},
        "notes": ["Per-run shared transform timings must not be summed once per classifier.",
                  "Ridge alpha=1 is illustrative; no hyperparameter search or cosine baseline is implemented."],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also protects against a concurrent run overwriting evidence.
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(report["status"])
    print(f"Completed {len(report['runs'])} classifier fits; report: {args.output}")


if __name__ == "__main__":
    main()
