import numpy as np
from skimage import io
from pathlib import Path


def save_image(img: np.ndarray, path: Path | str) -> None:
    """Saves a 2D numpy array to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ext = p.suffix.lower()

    if ext == ".npy":
        np.save(p, img)
    elif ext == ".npz":
        np.savez_compressed(p, img=img)
    else:
        # Safe format conversion for saving visual images if needed, or save raw tiff
        if ext in {".png", ".jpg", ".jpeg"}:
            # Basic linear stretch to [0, 255] for visual formats
            min_v, max_v = img.min(), img.max()
            if max_v > min_v:
                vis_img = ((img - min_v) / (max_v - min_v) * 255).astype(np.uint8)
            else:
                vis_img = np.zeros_like(img, dtype=np.uint8)
            io.imsave(str(p), vis_img, check_contrast=False)
        else:
            # Tiff saves float32 directly
            io.imsave(str(p), img.astype(np.float32), check_contrast=False)
