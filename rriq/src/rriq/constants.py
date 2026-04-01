"""Global constants and defaults for the RRIQ package."""

from pathlib import Path

DEFAULT_EPSILON = 1e-8
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy", ".npz"}

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "runs"
