"""Scripts to download or generate data."""

from pathlib import Path

from rl_wxai.config import DATA_RAW


def load_raw(filename: str) -> Path:
    """Return the full path to a raw data file."""
    return DATA_RAW / filename
