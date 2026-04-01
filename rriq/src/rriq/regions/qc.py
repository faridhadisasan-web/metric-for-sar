import numpy as np
from skimage.color import label2rgb
from matplotlib import pyplot as plt
import io
from pathlib import Path
from PIL import Image


def generate_qc_overlay(
    image: np.ndarray, mask: np.ndarray, path: Path | str, title: str = "Overlay"
) -> None:
    """Saves a visual overlay of a mask on the original image."""
    # Normalize base image for visualization
    base = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

    # Create colored overlay
    overlay = label2rgb(mask, image=base, bg_label=0, colors=["red"], alpha=0.3)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    buf.seek(0)
    im = Image.open(buf)
    im.save(Path(path))
