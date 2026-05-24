"""Download and place raw dataset into data/raw/.

Adapt this module once the dataset is chosen.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def download(url: str, dest: Path) -> None:
    """Download a file from url to dest.

    Args:
        url: Source URL.
        dest: Destination path.
    """
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # TODO: set dataset URL and filename once dataset is chosen
    # download("https://...", RAW_DIR / "interactions.csv")
    logger.info("Dataset download not yet configured — add URL to make_dataset.py")
