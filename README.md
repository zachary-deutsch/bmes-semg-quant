# BMES2025 sEMG: evaluating QUANT features

Research software and recovery notes for **sEMG Hand and Wrist Movement Classification: Evaluating QUANT Features**, by **Zachary Deutsch, Eren Tascesme, and Neta Rabin**. The research asks whether interval quantile features paired with machine learning classifiers can support hand and wrist movement classification from sEMG.

**Status: original code recovered; synthetic functionality verified; poster results have not been reproduced.** This source release contains no NinaPro recordings, fitted models, or recovered experimental results.

The [BMES poster](https://zacharydeutsch.com/20250908_BMES_Deutsch.pdf) is the research reference. Zachary is its first-listed author and supplied historical source for this project. Research credit is shared among the three authors; this repository does not assert sole ownership of the research. QUANT itself is upstream work by Dempster, Schmidt, and Webb, rather than a new algorithm attributed to the poster authors. The new package, tests, and documentation here were prepared with Codex on 2026-10-26.

## Run the small demo

From this repository, with **Python 3.14** (tested with 3.14.6 on macOS arm64):

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m bmes_semg.cli --synthetic
```

The last command is the single demo entry point. It generates 36 toy trials in memory (2 fictional subjects, 3 classes, 6 repetitions), computes QUANT features, and runs subject-specific ExtraTrees and an illustrative Ridge classifier. It writes a new, uniquely named JSON file under `outputs/`, including configuration, split identities, scores, environment, and timing. It never downloads data. No model is saved. Use `--output outputs/my_run.json` to choose a new destination; existing reports are protected from overwrite.

**Synthetic scores only verify that the software runs. They are not BMES results, physiological validation, clinical evidence, or an estimate of real sEMG accuracy.** The generator encodes deliberately simple amplitude/frequency differences. The demo uses 2 resamples and short 128-sample signals to keep it small.

```bash
python -m pytest -q
```

The final verification record is in [docs/verification.md](docs/verification.md); [outputs/synthetic_verified.json](outputs/synthetic_verified.json) is an actual saved smoke run, clearly labeled synthetic. Dependencies are pinned in `pyproject.toml`; exact installed package versions are captured in [docs/environment-lock.txt](docs/environment-lock.txt). This is a newly tested environment, not a recovered 2025 environment.

## Relationship to the research

The poster describes NinaPro DB7, QUANT with ExtraTrees (200 estimators), Ridge, 30 resamples, subject-specific 4 training / 1 validation / 1 test trial per class, and a separate cosine baseline.

The locally recovered July 26 script (not included in this release) is a historical variant: it combines QUANT with a flattened 12 x 12 cosine matrix, uses unseeded NumPy trial shuffling, assumes equal-length trials, and leaves one transform timing field at zero. No recovered file is confirmed to be the exact code that produced the poster. A July 27 proposed revision is recorded in the source history with no evidence it was executed.

The runnable package uses the recovered QUANT algorithm, deterministic balanced 4/1/1 splits, training-only standardization, explicit padding checks, and 200-tree ExtraTrees. Ridge alpha=1 is a new illustrative choice. The package does not implement the historical cosine mixture, the poster's separate cosine baseline, hyperparameter sweeps, or raw DB7 preprocessing. Those omissions and the full reconciliation are in [docs/methods.md](docs/methods.md).

## Real-data path

NinaPro recordings and the original preprocessing scripts are absent. [Download instructions and observed archive sizes](docs/data-access.md) explain how to obtain the official DB7 release manually. This repository does not download it automatically. Follow the official access terms and citation requirements; do not assume a license for dataset redistribution from the upstream code license. The loader accepts an already segmented, movement-only CSV with explicit trial identity:

```text
time,subject,class,trial,emg1,emg2,...,emg12
```

[data/README.md](data/README.md) defines the schema and checks that must be resolved before a real-data run. Once a locally prepared, reviewed one-subject CSV exists:

```bash
python -m bmes_semg.cli --csv data/one_subject.csv --config configs/real_data_check.json --output outputs/one_subject_check.json
```

This is a one-resample data integration check, with an explicit unverified-reproduction label. Full reproduction still needs the exact subject/class selection, segmentation and padding rules, classifier settings (especially Ridge), original seeds/splits or an agreed replication protocol, and comparison with original result tables. Thirty resamples are a later explicit configuration change after that audit; there is no automatic full benchmark or data download.

## What is where

| Path | Purpose |
| --- | --- |
| `archive/` | Attributed QUANT source, safe recovery dates and SHA-256 hashes |
| `src/bmes_semg/` | New validated loader, QUANT wrapper, evaluation, CLI |
| `configs/` | Small synthetic and one-resample local-data configurations |
| `tests/` | Focused integrity, split, preprocessing, feature parity, integration tests |
| `docs/` | Method reconciliation, verification, environment record |
| `third_party/` | Pinned upstream reference and complete GPLv3 license text |

Software in this release is licensed under **GNU GPL version 3**; see [LICENSE](LICENSE) and [licensing and attribution](LICENSE_STATUS.md). [Archive provenance](archive/README.md) preserves dates and hashes without private recovery records. Historical July experiment scripts are retained unchanged locally and excluded from this release; only attributed QUANT and the new tested package are included. A passing synthetic demo does not establish scientific reproduction.
