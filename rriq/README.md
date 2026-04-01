# RRIQ: Radiomic Ratio-Image Quality Index

**RRIQ: Radiomic Ratio-Image Quality Index for No-Clean-Reference SAR Despeckling Evaluation**

This repository provides a fully reproducible Python framework to evaluate SAR despeckling filters without requiring a clean reference image.

## Overview

RRIQ evaluates despeckling filters by:
1. Generating filtered versions of raw SAR images.
2. Computing the ratio image between the original noisy image and the filtered image.
3. Automatically identifying homogeneous, edge, and scatterer regions.
4. Extracting PyRadiomics features.
5. Computing classical SAR evaluation metrics (ENL, EPD-ROA, MOR, TCR, etc.).
6. Constructing an interpretable summary composite index, the **RRIQ**.

## Installation

Ensure you have Python 3.11.

```bash
git clone https://github.com/rriq-authors/rriq.git
cd rriq
pip install -e ".[dev]"
```

## Quickstart

Put your `.tif` SAR intensity images in `data/raw/` and run the full pipeline:

```bash
python -m rriq.cli run-all --config configs/default.yaml
```

The pipeline will save all logs, parameters, tables, metrics, and plots to `outputs/runs/<timestamp>/`.

## Folder layout

- `data/raw/`: Input original noisy SAR intensity images.
- `data/synthetic_clean/`: Optional clean synthetic data.
- `configs/`: YAML configuration files.
- `src/rriq/`: Source code.
- `outputs/runs/`: Output artifacts (metrics, plots, masks).

## Note on Optional Deep Filters
Deep filters (e.g., MERLIN) require the optional `[deep]` dependencies:
```bash
pip install -e ".[deep]"
```
