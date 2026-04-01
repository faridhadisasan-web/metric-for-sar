import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage.exposure import equalize_hist


def plot_triplet(
    original: np.ndarray, filtered: np.ndarray, ratio: np.ndarray, output_path: Path
) -> None:
    """Saves a panel showing the original, filtered, and ratio images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Use equalization for better display of SAR images
    axes[0].imshow(equalize_hist(original), cmap="gray")
    axes[0].set_title("Original (Noisy)")
    axes[0].axis("off")

    axes[1].imshow(equalize_hist(filtered), cmap="gray")
    axes[1].set_title("Filtered")
    axes[1].axis("off")

    # Ratio should ideally look like pure noise
    vmax = np.percentile(ratio, 99)
    vmin = np.percentile(ratio, 1)
    axes[2].imshow(ratio, cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title("Ratio Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
