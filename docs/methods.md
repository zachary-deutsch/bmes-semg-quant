# Method reconciliation and limits

Reference: [Deutsch, Tascesme, Rabin, BMES2025 poster](https://zacharydeutsch.com/20250908_BMES_Deutsch.pdf), inspected locally on 2026-10-26 by text extraction and page rendering. Historical code was inspected before execution of the operational QUANT copy. The original experiment scripts were not executed.

| Item | Poster | Recovered July 26 user script | New local package |
| --- | --- | --- | --- |
| Data | NinaPro DB7; 20 subjects, 12 channels | Prepared `all_trials.csv`, 12 channels; any subjects present | Synthetic 2-subject demo; optional reviewed local CSV |
| QUANT | Interval quantiles; varied depths | depth=2, div=4 defaults; recovered June QUANT import | Same algorithm, wrapper checks; scalar compatibility fix |
| ExtraTrees | 200 estimators, varied max features | 200, max_features=256, entropy | 200, max_features=32, entropy, one classifier worker |
| Ridge | Present, 30 resamples | Absent | alpha=1 demo only; not recovered poster settings |
| Split | Each class: 4 train, 1 validation, 1 test, subject-specific | 5-per-class pool split 80/20 with StratifiedShuffleSplit plus 1 test | Direct per-class seeded permutation into 4/1/1 |
| Resamples | 30 | Default 30 | Demo 2; local-data check 1 |
| Cosine | Separate ExtraTrees baseline; 66 features | All 144 matrix entries including diagonal and symmetric duplicates, concatenated with QUANT with 0.3/0.7 weights | Not implemented; no claimed parity with cosine experiment |
| Length handling | Not specified sufficiently to rebuild preprocessing | `np.stack` within subject; no padding | Fit right-padding length on train only, or use a predeclared length; reject longer held-out signals |
| Scaling | Not specified in poster | StandardScaler fitted on training QUANT and cosine blocks separately | Training-only StandardScaler on QUANT |
| Randomness | Exact seeds unavailable | NumPy global shuffle unseeded; classifier and inner split seed=resample index | Local NumPy generator seeded in config; same split across both classifiers |
| Timing | 8-core laptop; aggregate compute plots | Train-transform timer includes validation/test transforms; test-transform timer is never updated | Separate train/validation/test transform clocks and fit/predict clocks; no hardware comparison |

The June 1 implementation computes quantiles over raw, smoothed first difference, second difference, and FFT magnitude representations. Means are subtracted from alternating quantiles. Its four functions/classes match the pinned AALTD2024 code at the AST level. The operational copy changes only the tensor-scalar-to-integer conversion used in calculating interval shift, avoiding a modern NumPy deprecation warning; numerical parity tests cover several lengths and depths.

The July 26 script's 30 resamples and 200-tree defaults align with the poster, but its mixed features and lack of Ridge do not reproduce the poster's full experiment set. Seeded direct class splitting is a deliberate new protocol and will not recover historical trial assignments. Training examples may be held out in a different resample, as expected in repeated hold-out; splits are disjoint within each resample. Resamples are correlated, so their scores should not be treated as independent participants.

The locally inspected July 27 proposed revision removes cosine, caches splits, sets `gini` and `n_jobs=-1`, and loops `range(2,6)` (depths 2 through 5), despite a header suggesting 2 through 6. Its `transform_test` timer also remains zero. It reports test scores for every grid configuration; selecting a winner using those scores would contaminate evaluation. It is historical proposed code, not evidence of an executed benchmark. The new CLI runs one fixed configuration and performs no tuning.

The new summary averages resamples per subject, then weights subjects equally. Its reported standard deviation is the population standard deviation of subject means (`ddof=0`), not a confidence interval. The poster's uncertainty convention and full numerical tables have not been independently recovered. Timings include shared transform work reused across classifiers, so those fields should be counted once per subject/resample. JSON floating-point runtimes will vary between runs.

## Open scientific details

- The official [DB7 guide](https://ninapro.hevs.ch/instructions/DB7.html) describes 20 intact participants plus 2 amputees, whereas the poster reports 20 subjects. Confirm the actual IDs and exclusions; do not silently add the two amputees. It also directs users to cite Krasoulis et al. (2017).
- Exercise labels must remain globally distinct when exercises are combined. The CSV loader does not infer missing exercise identity, detect movement onset, discard rest, filter, resample, or construct trials from raw `.mat` arrays.
- Recover the exact segmentation, filtering, rest treatment, sampling units, sample ordering, zero-padding policy, and maximum lengths. Existing pre-padded CSVs require a provenance audit; the new train-only policy cannot remove an earlier leak.
- Confirm the poster-final QUANT version, depths, max-feature grid, Ridge normalization/alpha selection, cosine implementation, seeds, test-selection policy, and result aggregation before claiming reproduction.
- The poster reference entry [2] lists QUANT authors incorrectly. The [primary QUANT record](https://arxiv.org/abs/2308.00928) credits Angus Dempster, Daniel F. Schmidt, and Geoffrey I. Webb. The local poster has not been altered.

No claim of state-of-the-art performance, clinical utility, calibrated prosthetic control, or benchmark equivalence is established by this repository's tests. Poster accuracy and runtime values are not regenerated here.
