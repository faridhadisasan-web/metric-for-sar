import numpy as np
import scipy.io
from skimage import io
from pathlib import Path
from rriq.io.validators import validate_sar_image
from rriq.constants import SUPPORTED_EXTENSIONS


def read_image(
    path: Path | str, coerce_complex: bool = False, reduce_multichannel: str = "error"
) -> np.ndarray:
    """Reads an image from disk and validates its format."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension: {ext}. Supported: {SUPPORTED_EXTENSIONS}"
        )

    if ext in {".npy"}:
        img = np.load(p)
    elif ext in {".npz"}:
        arr_dict = np.load(p)
        # Assuming the first array is the image
        img = arr_dict[list(arr_dict.keys())[0]]
    elif ext in {".mat"}:
        mat = scipy.io.loadmat(str(p))
        # Find the first array-like object that isn't metadata
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No valid arrays found in {p}")
        img = mat[keys[0]]
    else:
        # Tiff, Png, Jpg
        img = io.imread(str(p))

    return validate_sar_image(
        img, coerce_complex=coerce_complex, reduce_multichannel=reduce_multichannel
    )
