"""Store useful variables and configuration."""

from pathlib import Path

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = REPO_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"

# Model paths
MODELS_DIR = REPO_ROOT / "models"

# Report / figure paths
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
