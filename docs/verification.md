# Verification record

The local implementation passed **19 focused tests** on 2026-10-26; `pip check` found no broken requirements. A standalone synthetic CLI run completed eight classifier fits. An independent repeat matched all split identities, feature counts, validation scores and test scores. Timings vary by run. The sanitized publication candidate passed all 19 tests in 2.41 seconds and reproduced the saved synthetic scores and split identities; the final code check also passed 19 tests in 2.01 seconds.

Tested environment: Python 3.14.6, macOS arm64; NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, PyTorch 2.14.1, pytest 9.1.1. `environment-lock.txt` records installed packages, including auxiliary PDF inspection tools not required by the package. No historical 2025 environment was recovered. Follow the README for a fresh virtual environment.

```bash
python -m pytest -q
python -m pip check
python -m bmes_semg.cli --synthetic --output outputs/my_smoke.json
```

The saved [synthetic report](../outputs/synthetic_verified.json) contains two fictional subjects, three toy classes, six repetitions per class, two resamples, 3,384 QUANT features, 200-tree ExtraTrees, and illustrative Ridge alpha=1. Both toy classifiers scored 1.0 on validation and test; these easy synthetic scores have no benchmark or physiological meaning. No trained model was saved.

## Checks covered

- Trial identity preserves adjacent same-class repetitions; malformed rows, noncontiguous duplicate blocks, rest-class IDs, nonfinite values, empty inputs and within-trial time resets are rejected.
- Padding and scaling are fitted on training trials. Longer held-out inputs are rejected. Changing held-out values leaves actual classifier-training features byte-for-byte unchanged.
- Each class contributes 4/1/1 disjoint trials within a subject. Seeded assignments repeat; duplicate keys and incorrect trial counts fail. A temporary synthetic CSV exercises loader-to-classifier integration.
- Operational QUANT matches the included historical upstream-derived code at lengths/depths (8,1), (8,6), (64,4), (128,2). Its scalar compatibility conversion removes modern NumPy warnings; parity tests inspect the historical-source warnings.
- The CLI emits explicitly labeled finite synthetic JSON and protects existing reports from overwrite. Included historical source hashes are checked against the manifest.

## Scientific limits

No real-data training, raw-to-CSV preprocessing, full NinaPro benchmark, 30-resample poster replication, hyperparameter sweep or cosine baseline was run. The exact historical processed CSV, final experimental settings, split seeds and result tables have not been recovered sufficiently to establish reproduction. The historical experiment scripts are not part of this release.

The release contains no dataset or cache, no fitted models, no private recovery URLs/identifiers, and no personal absolute paths. NinaPro access is documented manually; no automatic downloader is included. A future real-data check requires reviewed preprocessing, explicit physical trial identity, all six repetitions per selected class, and a resolved length policy before any reproduction claim.

## Publication preparation

A separate release tree was created from reviewed tracked files without copying private Git history. All 28 release files were inspected for private recovery links/identifiers, personal absolute paths, common credential patterns, data/model payloads, broken local documentation links and source-hash mismatches; no matches or failures were found. The complete GPLv3 text and upstream attribution are included. The public release scope includes the licensed QUANT source and newly prepared package; the two surrounding July experiment scripts remain unchanged and local. GitHub account identity was verified before any planned publication. The release was then published to [zachary-deutsch/bmes-semg-quant](https://github.com/zachary-deutsch/bmes-semg-quant). GitHub API verification confirmed public visibility, default branch `main`, and detected GPL-3.0 licensing on 2026-10-26. The published history starts from the sanitized source tree; the earlier private recovery commit was not copied or pushed. Documentation-only publication updates do not change the tested implementation.
