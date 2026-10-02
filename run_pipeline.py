"""Main execution CLI for Credit Card Fraud Detection (Attention Autoencoder).

Coordinates the common preprocessing pipeline, model training on legitimate
transactions, and held-out test evaluation with validation-selected threshold.
"""

import argparse
import sys
from pathlib import Path
import torch

from src.config import config
from src.preprocessing import run_common_preprocessing, check_dataset_exists
from src.dataset import create_dataloaders
from src.training.train_autoencoder import train_autoencoder
from src.evaluation.evaluate_autoencoder import evaluate_autoencoder_pipeline, load_trained_autoencoder
from src.utils import setup_logger

logger = setup_logger("MainPipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attention Autoencoder for Credit Card Fraud Detection (ICT-4442 Mini Project)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "preprocess", "train", "evaluate"],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=config.preprocessing.raw_data_path,
        help="Path to raw creditcard.csv dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.training.num_epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.training.batch_size,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.training.learning_rate,
        help="Learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        choices=["robust", "standard"],
        default="robust",
        help="Scaler type for feature preprocessing (default: robust)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config.preprocessing.raw_data_path = args.raw_path
    config.preprocessing.scaler_type = args.scaler
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr

    logger.info(f"Executing pipeline stage: [{args.stage.upper()}]")

    # Verify dataset existence before attempting to run
    if not check_dataset_exists(config.preprocessing.raw_data_path):
        logger.error(
            f"\nDataset file NOT found at: {config.preprocessing.raw_data_path.resolve()}\n"
            f"Please place 'creditcard.csv' into 'data/raw/' before running this pipeline.\n"
            f"Download source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        )
        sys.exit(1)

    # Stage: Preprocess
    if args.stage in ["preprocess", "all"]:
        logger.info("=== STEP 1: Running Common Preprocessing ===")
        preprocessed = run_common_preprocessing(config.preprocessing)
    else:
        # Load from saved preprocessed splits
        import numpy as np

        splits_path = config.preprocessing.processed_dir / "processed_splits.npz"
        if not splits_path.exists():
            logger.info("Processed splits not found. Running preprocessing first ...")
            preprocessed = run_common_preprocessing(config.preprocessing)
        else:
            data = np.load(splits_path)
            preprocessed = {
                "X_train": data["X_train"],
                "y_train": data["y_train"],
                "X_val": data["X_val"],
                "y_val": data["y_val"],
                "X_test": data["X_test"],
                "y_test": data["y_test"],
                "X_train_legit": data["X_train_legit"],
            }

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_legit=preprocessed["X_train_legit"],
        X_val=preprocessed["X_val"],
        y_val=preprocessed["y_val"],
        X_test=preprocessed["X_test"],
        y_test=preprocessed["y_test"],
        batch_size=config.training.batch_size,
    )

    # Stage: Train
    if args.stage in ["train", "all"]:
        logger.info("=== STEP 2: Training Attention Autoencoder (Legitimate samples only) ===")
        model, history = train_autoencoder(train_loader, val_loader, config)
    elif args.stage == "evaluate":
        device = torch.device(config.training.device)
        model = load_trained_autoencoder(config.training.model_save_path, device)

    # Stage: Evaluate
    if args.stage in ["evaluate", "all"]:
        logger.info("=== STEP 3: Evaluating Model & Selecting Validation Threshold ===")
        eval_results = evaluate_autoencoder_pipeline(model, val_loader, test_loader, config)
        logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
