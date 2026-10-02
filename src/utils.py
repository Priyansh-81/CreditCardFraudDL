"""Utility functions for logging, reproducibility, and filesystem management."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def setup_logger(name: str = "CreditCardDL", log_file: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a standard console and optional file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def ensure_directories(*paths: Path) -> None:
    """Ensure all specified directories exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: Path, indent: int = 4) -> None:
    """Save dictionary to a formatted JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file into a Python dictionary."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
