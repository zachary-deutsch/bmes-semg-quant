# How to obtain NinaPro DB7 yourself

**Data are not included, and the repository does not automatically download them.** Use the synthetic demo without obtaining DB7.

1. Open the [official NinaPro DB7 instructions](https://ninapro.hevs.ch/instructions/DB7.html).
2. Read the dataset description, access/usage terms, and citation instructions. The guide directs users to Krasoulis et al., [*Improved prosthetic hand control with concurrent use of myoelectric and inertial measurements* (2017)](https://doi.org/10.1186/s12984-017-0284-4).
3. Under the DB7 Data table, choose the desired subject's preprocessed-file link. For a first local check, use Subject 1 rather than downloading all subjects. The observed official [Subject 1 archive](https://ninapro.hevs.ch/files/DB7_Preproc/Subject_1.zip) was 1,507,392,596 bytes (about 1.51 GB) on 2026-10-26. It contains `S1_E1_A1.mat` (624,115,205 bytes extracted) and `S1_E2_A1.mat` (885,827,097 bytes extracted). Allow room for both the archive and extraction. Links, sizes, and access conditions may change; start from the official guide.
4. Download manually to an ignored local directory such as `data/`, and extract locally. Keep recordings and derived CSVs out of Git. No account or click-through acceptance was presented during the scoped public access check; this is an observation, not a grant of redistribution rights.
5. Verify and record file hashes, subject/exercise identity, corrected label arrays, and preprocessing before creating any evaluation CSV. The original processed `all_trials.csv` used in the historical experiments is **not** supplied by simply downloading DB7.

## Turning the release into input for this repository

The official guide describes 12-channel EMG sampled at 2 kHz and corrected `restimulus` and `rerepetition` arrays. The linked dataset paper describes Hampel filtering and post-hoc timing refinement for its offline recordings. The download path is explicitly labeled `DB7_Preproc`; do not describe it as untouched raw acquisition. Inspect the actual arrays and release documentation before applying any additional filter or relabeling.

To prepare a small integration check, a reviewer should select a single subject and a documented set of classes, use corrected label/repetition boundaries to retain each full physical movement trial, exclude rest, and preserve all six physical repetitions per selected class. Explicitly document any further segmentation, filtering, resampling, and padding. No raw-MAT converter has been implemented or verified here. Historical preprocessing and the exact poster-final subset are still unknown.

Export only the reviewed trials into the schema in [data/README.md](../data/README.md). Then run the documented one-resample CSV check. Keep labels from different exercises globally distinct if combining exercises. The new train-only padding policy rejects held-out trials longer than its training-derived length; resolve that policy explicitly without silently truncating signals or deriving a length from held-out data.

This route supports a future data integration check. It does not recreate the original CSV or establish reproduction of poster accuracy/runtime. No real-data evaluation is claimed.
