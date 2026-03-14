"""
Load assertion data with ground-truth assertiveness scores for probe training.

Uses epistemic-integrity/scibert-finetuning/data/train_data.csv which contains
(text, assertiveness, source) columns. Assertiveness scores are human-annotated
labels for training linear probes to predict displayed confidence from activations.
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def get_assertion_data_path() -> Path:
    """Return path to train_data.csv in epistemic-integrity."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "epistemic-integrity" / "scibert-finetuning" / "data" / "train_data.csv"


def load_assertion_data(
    data_path: Path | str | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 3407,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load assertion data and split into train/val/test.

    Args:
        data_path: Path to train_data.csv. Defaults to epistemic-integrity/data/train_data.csv.
        train_ratio: Fraction for training (default 0.8).
        val_ratio: Fraction for validation (default 0.1).
        test_ratio: Fraction for test (default 0.1).
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) with columns: text, assertiveness, source.
    """
    if data_path is None:
        data_path = get_assertion_data_path()
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Assertion data not found: {data_path}")

    df = pd.read_csv(data_path)
    if "text" not in df.columns or "assertiveness" not in df.columns:
        raise ValueError(f"Expected columns 'text' and 'assertiveness', got: {list(df.columns)}")

    # Drop rows with missing text or assertiveness
    df = df.dropna(subset=["text", "assertiveness"])
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_ratio, random_state=seed, shuffle=True
    )
    # Second split: train vs val (val_ratio of total = val_ratio / (1 - test_ratio) of train_val)
    val_size = val_ratio / (1 - test_ratio)
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=seed, shuffle=True
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def get_assertiveness_stats(df: pd.DataFrame) -> dict:
    """Return min, max, mean, std of assertiveness for normalization."""
    return {
        "min": df["assertiveness"].min(),
        "max": df["assertiveness"].max(),
        "mean": df["assertiveness"].mean(),
        "std": df["assertiveness"].std(),
    }
