"""Download and cache the learning datasets to ``data/raw/``.

STUB (Stage 0): the loaders are implemented in Stage 1
(``src/xgblearn/data/loaders.py``). This entrypoint will iterate the datasets
from the learning path and trigger their cached downloads:

  1. Breast Cancer Wisconsin (sklearn built-in)   -- Stage 1
  2. California Housing      (sklearn built-in)    -- Stage 2
  3. Adult / Census Income   (OpenML)              -- Stage 3
  4. Credit Card Fraud       (OpenML)              -- Stage 4

For now it just reports intent so `make` / the README quickstart don't hit a
missing-file error.
"""

from __future__ import annotations


def main() -> int:
    print(
        "download_data: not implemented yet (Stage 1 fills in "
        "src/xgblearn/data/loaders.py). No data downloaded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
