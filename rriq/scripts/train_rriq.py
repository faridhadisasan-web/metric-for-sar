import argparse
import pandas as pd
from pathlib import Path
from rriq.utils.logging_utils import logger
from rriq.modeling.train import train_supervised_rriq
from rriq.modeling.export import export_rriq_formula

def main():
    parser = argparse.ArgumentParser(description="Train Supervised RRIQ Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics.csv generated from pipeline")
    parser.add_argument("--target", type=str, required=True, help="Column in metrics.csv to use as target")
    parser.add_argument("--out_dir", type=str, default="models", help="Directory to save the trained model")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        return

    df = pd.read_csv(metrics_path)
    if args.target not in df.columns:
        logger.error(f"Target column '{args.target}' not found in {metrics_path}")
        return

    model_path = out_dir / "rriq_model.pkl"
    scaler_path = out_dir / "rriq_scaler.pkl"
    formula_path = out_dir / "rriq_formula.json"

    model, scaler, features, weights = train_supervised_rriq(df, args.target, model_path, scaler_path)
    if model is None:
        logger.error("Failed to train model.")
        return

    logger.info("Training complete.")
    logger.info(f"Learned weights: {weights}")

    export_rriq_formula(weights, formula_path)
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Formula saved to {formula_path}")

if __name__ == "__main__":
    main()
