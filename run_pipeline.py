"""Main execution CLI for Credit Card Fraud Detection (ICT-4442 Mini Project).

Coordinates the common preprocessing pipeline, model training, and held-out
test evaluation with a validation-selected threshold. Select the model with
--model: the Attention Autoencoder (trained on legitimate transactions only)
or the supervised MLP baseline (trained on the full labelled training set).
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

from src.config import config
from src.preprocessing import run_common_preprocessing, check_dataset_exists
from src.dataset import create_dataloaders, create_supervised_dataloaders
from src.training.train_autoencoder import train_autoencoder
from src.training.train_mlp import train_mlp, load_trained_mlp, resolve_device
from src.training.train_cnn1d import train_cnn1d, load_trained_cnn1d
from src.evaluation.evaluate_autoencoder import evaluate_autoencoder_pipeline, load_trained_autoencoder
from src.evaluation.evaluate_supervised import (
    predict_dataloader,
    find_optimal_threshold,
    evaluate_test_set,
    save_supervised_metrics,
)
from src.evaluation.compare_models import compare_models
from src.evaluation.error_analysis import run_error_analysis
from src.utils import setup_logger

logger = setup_logger("MainPipeline")

IMPLEMENTED_MODELS = ["autoencoder", "mlp", "cnn1d"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep Learning for Credit Card Fraud Detection (ICT-4442 Mini Project)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["autoencoder", "mlp", "cnn1d", "lstm", "all"],
        default="autoencoder",
        help="Model to run (default: autoencoder)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "preprocess", "train", "evaluate", "compare"],
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
        default=None,
        help="Number of training epochs (default: model config value, 40)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training and evaluation (default: model config value, 512)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for AdamW optimizer (default: model config value, 1e-3)",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        choices=["robust", "standard"],
        default="robust",
        help="Scaler type for feature preprocessing (default: robust)",
    )
    return parser.parse_args()


def load_preprocessed(stage: str) -> dict:
    """Run preprocessing, or load the saved splits (and scaler) when not preprocessing."""
    if stage in ["preprocess", "all"]:
        logger.info("=== STEP 1: Running Common Preprocessing ===")
        return run_common_preprocessing(config.preprocessing)

    splits_path = config.preprocessing.processed_dir / "processed_splits.npz"
    if not splits_path.exists():
        logger.info("Processed splits not found. Running preprocessing first ...")
        return run_common_preprocessing(config.preprocessing)

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
    scaler_path = config.preprocessing.processed_dir / f"{config.preprocessing.scaler_type}_scaler.joblib"
    if scaler_path.exists():
        import joblib

        preprocessed["scaler"] = joblib.load(scaler_path)
    return preprocessed


def run_autoencoder(preprocessed: dict, stage: str) -> None:
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_legit=preprocessed["X_train_legit"],
        X_val=preprocessed["X_val"],
        y_val=preprocessed["y_val"],
        X_test=preprocessed["X_test"],
        y_test=preprocessed["y_test"],
        batch_size=config.training.batch_size,
    )

    if stage in ["train", "all"]:
        logger.info("=== STEP 2: Training Attention Autoencoder (Legitimate samples only) ===")
        model, history = train_autoencoder(train_loader, val_loader, config)
    elif stage == "evaluate":
        device = torch.device(config.training.device)
        model = load_trained_autoencoder(config.training.model_save_path, device)

    if stage in ["evaluate", "all"]:
        logger.info("=== STEP 3: Evaluating Model & Selecting Validation Threshold ===")
        evaluate_autoencoder_pipeline(model, val_loader, test_loader, config)


def run_mlp(preprocessed: dict, stage: str) -> None:
    t_cfg = config.mlp_training
    train_loader, val_loader, test_loader = create_supervised_dataloaders(
        X_train=preprocessed["X_train"],
        y_train=preprocessed["y_train"],
        X_val=preprocessed["X_val"],
        y_val=preprocessed["y_val"],
        X_test=preprocessed["X_test"],
        y_test=preprocessed["y_test"],
        batch_size=t_cfg.batch_size,
        num_workers=t_cfg.num_workers,
    )
    device = resolve_device(t_cfg.device)

    if stage in ["train", "all"]:
        logger.info("=== STEP 2: Training MLP Baseline (full labelled training set, pos_weight loss) ===")
        model, history = train_mlp(train_loader, val_loader, config)
    elif stage == "evaluate":
        model = load_trained_mlp(t_cfg.model_save_path, device)

    if stage in ["evaluate", "all"]:
        logger.info("=== STEP 3: Selecting Threshold on Validation & Evaluating Test Set ===")
        model = model.to(device)
        val_probs, y_val = predict_dataloader(model, val_loader, device)
        threshold_results = find_optimal_threshold(val_probs, y_val, metric=t_cfg.threshold_metric)
        threshold = threshold_results["optimal_threshold"]

        test_results = evaluate_test_set(model, test_loader, threshold, device, model_name="MLP Baseline")
        test_probs = test_results.pop("test_probs")
        y_test = test_results.pop("y_test")

        save_supervised_metrics(
            {
                "model": "MLP Baseline",
                "architecture": {
                    "input_dim": config.mlp_model.input_dim,
                    "hidden_dims": list(config.mlp_model.hidden_dims),
                    "dropout": config.mlp_model.dropout,
                    "use_batch_norm": config.mlp_model.use_batch_norm,
                },
                "pos_weight": t_cfg.pos_weight,
                "validation_selection": threshold_results,
                "test_metrics": test_results,
            },
            t_cfg.metrics_save_path,
        )
        np.savez_compressed(
            t_cfg.predictions_save_path,
            y_test=y_test,
            test_probs=test_probs,
            test_predictions=(test_probs >= threshold).astype(int),
            threshold=threshold,
        )

        logger.info("=== STEP 4: Error Analysis & Result Visualizations ===")
        run_error_analysis(
            preprocessed["X_test"],
            y_test,
            test_probs,
            threshold,
            scaler=preprocessed.get("scaler"),
            model_name="MLP Baseline",
        )


def run_cnn1d(preprocessed: dict, stage: str) -> None:
    t_cfg = config.cnn1d_training
    m_cfg = config.cnn1d_model
    train_loader, val_loader, test_loader = create_supervised_dataloaders(
        X_train=preprocessed["X_train"],
        y_train=preprocessed["y_train"],
        X_val=preprocessed["X_val"],
        y_val=preprocessed["y_val"],
        X_test=preprocessed["X_test"],
        y_test=preprocessed["y_test"],
        batch_size=t_cfg.batch_size,
        num_workers=t_cfg.num_workers,
    )
    device = resolve_device(t_cfg.device)

    if stage in ["train", "all"]:
        logger.info("=== STEP 2: Training 1-D CNN (full labelled training set, pos_weight loss) ===")
        model, history = train_cnn1d(train_loader, val_loader, config)
    elif stage == "evaluate":
        model = load_trained_cnn1d(t_cfg.model_save_path, device)

    if stage in ["evaluate", "all"]:
        logger.info("=== STEP 3: Selecting Threshold on Validation & Evaluating Test Set ===")
        model = model.to(device)
        val_probs, y_val = predict_dataloader(model, val_loader, device)
        threshold_results = find_optimal_threshold(val_probs, y_val, metric=t_cfg.threshold_metric)
        threshold = threshold_results["optimal_threshold"]

        test_results = evaluate_test_set(model, test_loader, threshold, device, model_name="1-D CNN")
        test_probs = test_results.pop("test_probs")
        y_test = test_results.pop("y_test")

        save_supervised_metrics(
            {
                "model": "1-D CNN",
                "architecture": {
                    "input_dim": m_cfg.input_dim,
                    "channels": list(m_cfg.channels),
                    "kernel_size": m_cfg.kernel_size,
                    "dropout": m_cfg.dropout,
                    "dense_dim": m_cfg.dense_dim,
                },
                "pos_weight": t_cfg.pos_weight,
                "validation_selection": threshold_results,
                "test_metrics": test_results,
            },
            t_cfg.metrics_save_path,
        )
        np.savez_compressed(
            t_cfg.predictions_save_path,
            y_test=y_test,
            test_probs=test_probs,
            test_predictions=(test_probs >= threshold).astype(int),
            threshold=threshold,
        )

        logger.info("=== STEP 4: Error Analysis & Result Visualizations ===")
        run_error_analysis(
            preprocessed["X_test"],
            y_test,
            test_probs,
            threshold,
            scaler=preprocessed.get("scaler"),
            model_name="1-D CNN",
        )


def main():
    args = parse_args()

    # Stage: compare alone does not require dataset loading
    if args.stage == "compare":
        logger.info("=== STEP: Running Master Cross-Model Comparison Harness ===")
        compare_models()
        return

    config.preprocessing.raw_data_path = args.raw_path
    config.preprocessing.scaler_type = args.scaler
    for t_cfg in (config.training, config.mlp_training, config.cnn1d_training):
        if args.epochs is not None:
            t_cfg.num_epochs = args.epochs
        if args.batch_size is not None:
            t_cfg.batch_size = args.batch_size
        if args.lr is not None:
            t_cfg.learning_rate = args.lr

    models = IMPLEMENTED_MODELS if args.model == "all" else [args.model]
    unavailable = [m for m in models if m not in IMPLEMENTED_MODELS]
    if unavailable:
        logger.error(f"Model(s) not implemented yet: {unavailable}. Available: {IMPLEMENTED_MODELS}")
        sys.exit(1)

    logger.info(f"Executing pipeline stage: [{args.stage.upper()}] for model(s): {models}")

    # Verify dataset existence before attempting to run
    if not check_dataset_exists(config.preprocessing.raw_data_path):
        logger.error(
            f"\nDataset file NOT found at: {config.preprocessing.raw_data_path.resolve()}\n"
            f"Please place 'creditcard.csv' into 'data/raw/' before running this pipeline.\n"
            f"Download source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        )
        sys.exit(1)

    preprocessed = load_preprocessed(args.stage)
    if args.stage == "preprocess":
        logger.info("Preprocessing completed successfully!")
        return

    for model_name in models:
        if model_name == "autoencoder":
            run_autoencoder(preprocessed, args.stage)
        elif model_name == "mlp":
            run_mlp(preprocessed, args.stage)
        elif model_name == "cnn1d":
            run_cnn1d(preprocessed, args.stage)

    if args.stage in ["evaluate", "all"]:
        logger.info("=== STEP 5: Updating Master Cross-Model Comparison Table & Curves ===")
        compare_models()

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
