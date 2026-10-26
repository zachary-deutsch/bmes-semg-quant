# Source history

Historical source was recovered on 2026-10-26 from project records. Private recovery links, conversation identifiers, surrounding messages, and personal paths are excluded from this source release. No archived version is confirmed to be the exact code that produced the poster.

| Source date | Source | Included? |
| --- | --- | --- |
| 2025-06-01 | `2025-06-01_quant.py`: QUANT implementation supplied for project recovery, attributed to Dempster, Schmidt, Webb and the multivariate upstream update | Yes; GPLv3 upstream algorithm, unchanged code, 3,969 UTF-8 bytes |
| 2025-07-26 | `run_alltrials_quant_singlecore.py`: QUANT + cosine, ExtraTrees, subject-specific 4/1/1 trials | No; retained in local recovery repository, outside public release scope |
| 2025-07-27 | Proposed QUANT-only cached-split grid-search revision | No; retained locally, outside public release scope; execution unconfirmed |

`manifest.json` records safe source dates and SHA-256 digests, including excluded historical variants for technical provenance. The included QUANT source preserves its original whitespace and absence of a trailing newline. Its digest independently matched the source recovery record; tests enforce the digest. The functions/classes match the pinned [AALTD2024 upstream source](https://github.com/angus924/aaltd2024/blob/b6baa47774bfe3dd0d1dc3febb1545928e288f3f/code/quant.py) at the AST level.

Historical experiment scripts remain in the private local recovery workspace. Their method differences are documented in [methods.md](../docs/methods.md). Their content and private recovery history are not part of this public package.

The included QUANT file was inspected before execution: it contains numerical imports and definitions, with no file I/O, shell execution, networking, credential access, or deletion. The operational copy in `src/bmes_semg/_quant.py` retains author attribution and marks its 2026 compatibility conversion. Other modules are new Codex-assisted wrapper/evaluation code. The complete upstream reference in `third_party/` is retained for attribution and comparison; its classifier is not executed by the demo.
