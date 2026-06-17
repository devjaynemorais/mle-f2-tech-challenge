"""Download RetailRocket dataset from Kaggle to data/raw/.

Usage:
    python scripts/download_dataset.py

Requires ~/.kaggle/kaggle.json with your Kaggle API credentials.
Get it at: https://www.kaggle.com/settings -> API -> Create New Token
"""

import sys
from pathlib import Path

DATASET = "retailrocket/ecommerce-dataset"
DEST = Path("data/raw")
EXPECTED_FILES = [
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
    "category_tree.csv",
]


def authenticate() -> None:
    """Authenticate with Kaggle API or exit with helpful message."""
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        print("ERROR: ~/.kaggle/kaggle.json not found.")
        print("Get it at: https://www.kaggle.com/settings -> API -> Create New Token")
        sys.exit(1)


def download() -> None:
    """Download and extract the RetailRocket dataset to data/raw/."""
    import kaggle

    DEST.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    print(f"Downloading {DATASET} ...")
    kaggle.api.dataset_download_files(DATASET, path=str(DEST), unzip=True)
    print(f"Extracted to {DEST}/")


def verify() -> bool:
    """Check expected files are present and print sizes."""
    all_ok = True
    for fname in EXPECTED_FILES:
        fpath = DEST / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1_048_576
            print(f"  OK  {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING  {fname}")
            all_ok = False
    return all_ok


def main() -> None:
    """Run download and verification."""
    authenticate()
    download()
    if not verify():
        print("\nSome files are missing — check the download.")
        sys.exit(1)
    print("\nDataset ready. Next: dvc add data/raw/ && dvc push")


if __name__ == "__main__":
    main()
