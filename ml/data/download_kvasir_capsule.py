#!/usr/bin/env python3
"""Download / register Kvasir-Capsule dataset.

Kvasir-Capsule is distributed via official channels (Simula / OSF). This script
documents the expected layout and can optionally fetch a URL if KVASIR_URL is set.

License: follow the dataset authors' terms. Do not redistribute restricted media
in this repository.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "kvasir-capsule"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("KVASIR_URL", ""))
    parser.add_argument("--out", type=Path, default=RAW)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    readme = args.out / "DOWNLOAD.md"
    readme.write_text(
        "# Kvasir-Capsule\n\n"
        "1. Obtain the dataset from the official Simula/OSF release.\n"
        "2. Extract so that labeled folders (or metadata CSV) live under this directory.\n"
        "3. Set `KVASIR_URL` and re-run this script only if you have a direct zip URL.\n"
        "4. Then run `python ml/data/preprocess.py` and `python ml/data/split_by_video.py`.\n\n"
        "Citation: Smedsrud et al., Kvasir-Capsule, Scientific Data, 2021.\n",
        encoding="utf-8",
    )

    if not args.url:
        print(f"No KVASIR_URL provided. Wrote instructions to {readme}")
        print(f"Expected extract path: {args.out}")
        return

    zip_path = args.out / "kvasir-capsule.zip"
    print(f"Downloading {args.url} -> {zip_path}")
    urllib.request.urlretrieve(args.url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(args.out)
    print("Download complete.")


if __name__ == "__main__":
    main()
