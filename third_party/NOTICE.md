# QUANT attribution

QUANT: Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb, *QUANT: A Minimalist Interval Method for Time Series Classification*, [arXiv:2308.00928](https://arxiv.org/abs/2308.00928).

Multivariate source reference: Angus Dempster, Chang Wei Tan, Lynn Miller, Navid Mohammadi Foumani, Daniel F. Schmidt, Geoffrey I. Webb, *Highly Scalable Time Series Classification for Very Large Datasets*, AALTD 2024. [Upstream repository](https://github.com/angus924/aaltd2024).

Pinned commit: `b6baa47774bfe3dd0d1dc3febb1545928e288f3f`, retrieved 2026-10-26. The unmodified reference file is `aaltd2024_quant_reference.py`; it is not the package's import path and is not run. The upstream `LICENSE` is stored verbatim as `aaltd2024_LICENSE.txt` (GNU GPL version 3). `upstream.json` and the archive manifest record the reference.

The recovered June 1 code's `make_intervals`, `f_quantile`, `IntervalModel`, and `Quant` definitions are AST-identical to these upstream definitions. The operational copy's only functional edit is conversion of the tensor interval-count scalar to an integer before NumPy arithmetic, avoiding a deprecation warning. This change was made with Codex on 2026-10-26 and is tested against the recovered original. New wrapper and evaluation code is identified separately; no grant of other authors' rights is implied.
