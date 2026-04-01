import numpy as np
from typing import List, Tuple


def extract_patches(
    img: np.ndarray, patch_size: int = 256, stride: int = 128
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """Extracts patches from an image. Returns list of (patch, (row_start, col_start))."""
    h, w = img.shape
    patches = []

    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            patch = img[r : r + patch_size, c : c + patch_size]
            patches.append((patch, (r, c)))

    return patches
