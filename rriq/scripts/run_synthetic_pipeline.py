import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rich.console import Console

from rriq.utils.logging_utils import logger
from rriq.utils.config import load_config
from rriq.utils.paths import get_run_dir

from rriq.io.manifest import build_manifest
from rriq.io.readers import read_image
from rriq.io.writers import save_image

from rriq.modeling.synthetic_mode import generate_synthetic_degradations
from rriq.preprocessing.transforms import apply_transforms
from rriq.regions.masks import generate_all_masks
from rriq.radiomics.extractor import extract_features
from rriq.radiomics.aggregations import flatten_features
from rriq.preprocessing.scaling import scale_to_255

console = Console()

def run_synthetic_pipeline(config_path: str | Path):
    """Executes the synthetic monotonicity mode."""
    start_time = time.time()
    cfg = load_config(config_path)

    io_cfg = cfg.get("io", {})
    run_dir = get_run_dir(io_cfg.get("output_dir", "outputs/runs"), name="synthetic_run")

    synthetic_dir = Path("data/synthetic_clean")
    if not synthetic_dir.exists() or not any(synthetic_dir.iterdir()):
        logger.warning(f"No files found in {synthetic_dir}. Skipping synthetic mode.")
        return None

    logger.info(f"Starting Synthetic Monotonicity Pipeline. Output dir: {run_dir}")
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    manifest_path = run_dir / "synthetic_manifest.csv"
    manifest = build_manifest(synthetic_dir, manifest_path, io_cfg.get("coerce_complex_to_magnitude", False), io_cfg.get("reduce_multichannel", "error"))

    look_ladder = cfg.get("synthetic", {}).get("looks", [1, 2, 4, 8, 16])

    all_metrics = []

    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing synthetic targets"):
        img_id = row["image_id"]
        img_path = Path(row["file_path"])

        raw_clean = read_image(img_path)
        clean_img = apply_transforms(raw_clean, cfg.get("preprocessing", {}))

        # 1. Generate Degradations
        degradations = generate_synthetic_degradations(clean_img, look_ladder)

        # 2. Extract features across degradations to measure monotonicity
        for deg_name, deg_data in degradations.items():
            noisy_img = deg_data["image"]
            severity = deg_data["severity_target"]

            # Save generated noisy image
            img_out_dir = run_dir / "images" / img_id
            img_out_dir.mkdir(parents=True, exist_ok=True)
            save_image(noisy_img, img_out_dir / f"{deg_name}.tif")

            # Use Noisy image to build masks (representing reality where we only have noisy)
            masks = generate_all_masks(noisy_img, cfg.get("masks", {}))

            # For Monotonicity, the "Filter" is just the identity or we treat the Noisy image as the target
            # So the ratio is clean/noisy or we just analyze the noisy directly.
            # In SAR despeckling evaluation, Monotonicity Mode evaluates how features respond to the NOISE level itself.
            # So we extract features from the Noisy image.

            row_metrics = {
                "image_id": img_id,
                "degradation": deg_name,
                "looks": deg_data["looks"],
                "severity_target": severity
            }

            # Since there's no filter applied yet, ratio is 1. We just extract radiomics on the noisy image itself.
            # to see if radiomics monotonically track the noise severity.
            if cfg.get("pipeline", {}).get("run_radiomics", True):
                scaled_noisy = scale_to_255(noisy_img)
                rad_cfg_filtered = "configs/radiomics/pyradiomics_filtered.yaml"

                for m_name, mask_arr in masks.items():
                    f_feats = extract_features(scaled_noisy, mask_arr, rad_cfg_filtered)
                    row_metrics.update(flatten_features(f_feats, prefix=f"noisy_{m_name}"))

            all_metrics.append(row_metrics)

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(metrics_dir / "synthetic_monotonicity.csv", index=False)

        # Calculate monotonicity (Spearman correlation) for all extracted features
        from rriq.modeling.monotonicity import screen_monotonic_features
        feat_cols = [c for c in df_metrics.columns if c.startswith("noisy_original_")]

        monotonic_feats = screen_monotonic_features(df_metrics, feat_cols, "severity_target", threshold=0.7)

        pd.DataFrame({"monotonic_features": monotonic_feats}).to_csv(metrics_dir / "monotonic_features.csv", index=False)

        logger.info(f"Pipeline completed. Found {len(monotonic_feats)} highly monotonic features.")
        return df_metrics
    return None

if __name__ == "__main__":
    import sys
    run_synthetic_pipeline(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")
