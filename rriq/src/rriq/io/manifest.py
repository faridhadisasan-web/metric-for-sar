import pandas as pd
from pathlib import Path
from rriq.io.readers import read_image
from rriq.constants import SUPPORTED_EXTENSIONS
from rriq.utils.logging_utils import logger


def build_manifest(
    input_dir: Path | str,
    output_csv: Path | str,
    coerce_complex: bool = False,
    reduce_multichannel: str = "error",
) -> pd.DataFrame:
    """Scans a directory for images, collects metadata, and saves a manifest CSV."""
    in_dir = Path(input_dir)
    out_csv = Path(output_csv)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    records = []

    for file_path in in_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                img = read_image(
                    file_path,
                    coerce_complex=coerce_complex,
                    reduce_multichannel=reduce_multichannel,
                )
                h, w = img.shape
                records.append(
                    {
                        "image_id": file_path.stem,
                        "file_path": str(file_path.absolute()),
                        "width": w,
                        "height": h,
                        "dtype": str(img.dtype),
                        "min": float(img.min()),
                        "max": float(img.max()),
                        "mean": float(img.mean()),
                        "std": float(img.std()),
                        "source_name": file_path.parent.name,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

    df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Manifest built with {len(df)} entries at {out_csv}")
    return df
