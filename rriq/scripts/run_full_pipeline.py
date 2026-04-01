import sys
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rich.console import Console

from rriq.utils.logging_utils import logger
from rriq.utils.config import load_config, save_config
from rriq.utils.paths import get_run_dir

from rriq.io.manifest import build_manifest
from rriq.io.readers import read_image
from rriq.io.writers import save_image

from rriq.preprocessing.transforms import apply_transforms
from rriq.filters.registry import get_filter
from rriq.ratio.compute import compute_ratio, compute_log_ratio

from rriq.regions.masks import generate_all_masks
from rriq.regions.qc import generate_qc_overlay

from rriq.radiomics.extractor import extract_features
from rriq.radiomics.aggregations import flatten_features
from rriq.metrics.composite_baselines import compute_all_baselines

from rriq.modeling.calibrate import build_unsupervised_rriq
from rriq.modeling.export import export_rriq_formula

from rriq.visualization.qc_panels import plot_triplet
from rriq.reporting.markdown_report import generate_markdown_report
from rriq.preprocessing.scaling import scale_to_255

console = Console()

def run_pipeline(config_path: str | Path):
    """Executes the full RRIQ evaluation pipeline."""
    start_time = time.time()
    cfg = load_config(config_path)

    io_cfg = cfg.get("io", {})
    run_dir = get_run_dir(io_cfg.get("output_dir", "outputs/runs"))

    logger.info(f"Starting RRIQ pipeline. Output dir: {run_dir}")
    save_config(cfg, run_dir / "config.yaml")

    # Dirs
    metrics_dir = run_dir / "metrics"
    vis_dir = run_dir / "visualizations"
    metrics_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    # 1. Manifest
    input_dir = Path(io_cfg.get("input_dir", "data/raw"))
    manifest_path = run_dir / "manifest.csv"

    if not input_dir.exists() or not any(input_dir.iterdir()):
        logger.warning(f"No files found in {input_dir}. Exiting.")
        return None

    manifest = build_manifest(input_dir, manifest_path, io_cfg.get("coerce_complex_to_magnitude", False), io_cfg.get("reduce_multichannel", "error"))

    all_metrics = []

    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing images"):
        img_id = row["image_id"]
        img_path = Path(row["file_path"])

        # 2. Read & Preprocess
        raw_img = read_image(img_path)
        img = apply_transforms(raw_img, cfg.get("preprocessing", {}))

        # Save Original QC
        img_out_dir = run_dir / "images"
        img_out_dir.mkdir(exist_ok=True)
        save_image(img, img_out_dir / f"{img_id}_original.tif")

        # 3. Build Masks
        masks = generate_all_masks(img, cfg.get("masks", {}))

        for m_name, mask_arr in masks.items():
            mask_dir = run_dir / "masks" / img_id
            mask_dir.mkdir(parents=True, exist_ok=True)
            save_image(mask_arr * 255, mask_dir / f"{m_name}.png")
            generate_qc_overlay(img, mask_arr, mask_dir / f"{m_name}_overlay.png")

        # 4. Filters & Ratios & Metrics
        filters_cfg = cfg.get("filters", {}).get("classical", [])

        for f_cfg in filters_cfg:
            f_name = f_cfg["name"]
            f_params = f_cfg.get("params", {})
            f_func = get_filter(f_name)

            filtered = f_func(img, **f_params)
            ratio = compute_ratio(img, filtered, eps=io_cfg.get("eps", 1e-8))

            # QC Visual
            plot_triplet(img, filtered, ratio, vis_dir / f"{img_id}_{f_name}_panel.png")

            row_metrics = {
                "image_id": img_id,
                "filter_name": f_name,
                "filter_params": str(f_params)
            }

            # Baselines
            if cfg.get("pipeline", {}).get("run_baselines", True):
                b_metrics = compute_all_baselines(img, filtered, ratio, masks)
                row_metrics.update(b_metrics)

            # Radiomics Extraction
            if cfg.get("pipeline", {}).get("run_radiomics", True):
                # We need radiomics scaled to [0, 255] for reproducible binning
                scaled_filtered = scale_to_255(filtered)

                # We can use either log ratio or standard ratio
                if cfg.get("ratio", {}).get("compute_log_ratio", True):
                    ratio_to_scale = compute_log_ratio(ratio, eps=io_cfg.get("eps", 1e-8))
                else:
                    ratio_to_scale = ratio
                scaled_ratio = scale_to_255(ratio_to_scale)

                rad_cfg_filtered = "configs/radiomics/pyradiomics_filtered.yaml"
                rad_cfg_ratio = "configs/radiomics/pyradiomics_ratio.yaml"

                # Extract for each mask
                for m_name, mask_arr in masks.items():
                    # Filtered image features
                    f_feats = extract_features(scaled_filtered, mask_arr, rad_cfg_filtered)
                    row_metrics.update(flatten_features(f_feats, prefix=f"filtered_{m_name}"))

                    # Ratio image features
                    r_feats = extract_features(scaled_ratio, mask_arr, rad_cfg_ratio)
                    row_metrics.update(flatten_features(r_feats, prefix=f"ratio_{m_name}"))

            all_metrics.append(row_metrics)

    # 5. RRIQ Calculation & Reporting
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)

        if cfg.get("pipeline", {}).get("run_rriq", True):
            weights = cfg.get("rriq", {}).get("weights", {
                "residual_purity": 0.35,
                "structure_preservation": 0.30,
                "radiometric_preservation": 0.20,
                "scatterer_preservation": 0.15
            })
            df_metrics = build_unsupervised_rriq(df_metrics, weights)
            export_rriq_formula(weights, metrics_dir / "rriq_formula.json")

        df_metrics.to_csv(metrics_dir / "metrics.csv", index=False)
        generate_markdown_report(run_dir, cfg, df_metrics)
        logger.info(f"Pipeline completed in {time.time() - start_time:.2f}s")
        return df_metrics
    return None

if __name__ == "__main__":
    run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")
