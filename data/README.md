# Local data only

No NinaPro files are included or retained. The historical prepared CSV is not included. See [manual download instructions](../docs/data-access.md) for the official source and observed size. Observe its terms and cite its dataset paper, Krasoulis et al. (2017). Public accessibility is not treated as a redistribution license.

For the smallest useful real-data check, provide **one reviewed subject** with at least two movement classes, exactly six trials per class, and 12 channels. Prefer the original prepared CSV plus preprocessing code/provenance. A new raw-to-CSV converter is intentionally absent because the historical segmentation rules have not been recovered.

The loader requires unique headers `time,subject,class,trial,emg1,...,emg12`; extra named columns are ignored. IDs must be positive integers (class 0/rest is rejected); signals and times must be finite numeric values. Each `(subject,class,trial)` key must occupy one contiguous block and must be globally unique within the file. Time must strictly increase within each block; it may restart in the next block. Channel ordering is emg1 through emg12, and signals are transposed to channels x samples. Original units must be documented externally; this loader does not convert them. Do not assign new trial IDs to windows from the same physical trial and then split them as independent trials.

Each trial can initially have its own length. Padding length is fitted from the training trials unless `pad_length` is explicitly declared in the config before inspecting held-out signals. Longer validation/test trials raise an error; no truncation or hidden recalculation is performed. A pre-padded historical CSV should be audited for how its length was chosen. The QUANT wrapper requires a padded length of at least eight samples.

Before a run, record subject IDs, movement label mapping across exercises, source file digests, corrected stimulus/repetition fields used, rest removal, movement detection, sample rate, filtering, and padding. The poster reports 20 subjects, but the official DB7 guide describes 20 intact participants and two amputees; the exact study subset remains unconfirmed.

Keep recordings and derived CSVs here or outside the repository. `.gitignore` excludes this directory's payloads and common raw data/model formats throughout the repository; it is a convenience, not a substitute for checking what is staged before any future release. The CLI reads local files only and records a SHA-256 digest without recording the absolute input path in its output.
