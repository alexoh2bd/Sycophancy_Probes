"""
Train ridge regression probes per (layer, head) to predict assertiveness from MHA activations.

Uses ground-truth assertiveness scores from epistemic-integrity train_data.csv.
Saves probe weights and R²/MSE per head for heatmap visualization.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Paths
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROBE_DIR = _REPO_ROOT / "probe"
if str(_PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBE_DIR))

from utils import load_model


def get_model_config(model):
    """Extract num_layers, num_heads, head_dim from model."""
    if "gemma" in str(type(model)).lower():
        cfg = model.config.text_config
    else:
        cfg = model.config
    return {
        "num_layers": cfg.num_hidden_layers,
        "num_heads": cfg.num_attention_heads,
        "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
    }


def train_probes(
    activations: np.ndarray,
    assertiveness: np.ndarray,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    alpha: float = 1.0,
) -> tuple[dict, dict]:
    """
    Train ridge regression probe for each (layer, head).

    Args:
        activations: (n_samples, n_layers, n_heads, head_dim).
        assertiveness: (n_samples,) target values.
        num_layers, num_heads, head_dim: Model config.
        alpha: Ridge regularization strength.

    Returns:
        (r2_dict, mse_dict, probe_weights) mapping "layer_head" -> metric/weights.
    """
    from sklearn.metrics import r2_score, mean_squared_error

    r2_dict = {}
    mse_dict = {}
    probe_weights = {}

    for layer in tqdm(range(num_layers), desc="Training probes"):
        for head in range(num_heads):
            key = f"{layer}_{head}"
            X = activations[:, layer, head, :]  # (n, head_dim)
            y = assertiveness

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            ridge = Ridge(alpha=alpha, random_state=3407)
            ridge.fit(X_scaled, y)
            y_pred = ridge.predict(X_scaled)

            r2_dict[key] = float(r2_score(y, y_pred))
            mse_dict[key] = float(mean_squared_error(y, y_pred))
            probe_weights[key] = {
                "coef": ridge.coef_.astype(np.float32),
                "intercept": float(ridge.intercept_),
                "scaler_mean": scaler.mean_.astype(np.float32),
                "scaler_scale": scaler.scale_.astype(np.float32),
            }

    return r2_dict, mse_dict, probe_weights


def run_full_pipeline(
    model_id: str = "gemma-3",
    data_path: Path | None = None,
    output_dir: Path | str = "calibration_output",
    max_samples: int | None = None,
    alpha: float = 1.0,
):
    """
    End-to-end: load data, extract activations, train probes, save results.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from calibration.data.assertion_loader import load_assertion_data
    from calibration.probing.extract_activations import extract_activations_batch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, val_df, test_df = load_assertion_data(data_path=data_path)
    if max_samples:
        train_df = train_df.head(max_samples)
    texts = train_df["text"].tolist()
    assertiveness = train_df["assertiveness"].values.astype(np.float32)

    # Load model and extract activations
    model, processor = load_model(model_id)
    activations = extract_activations_batch(
        model, processor, texts, model_id, max_length=2048
    )
    activations_np = activations.numpy()

    # Train probes
    cfg = get_model_config(model)
    r2_dict, mse_dict, probe_weights = train_probes(
        activations_np,
        assertiveness,
        cfg["num_layers"],
        cfg["num_heads"],
        cfg["head_dim"],
        alpha=alpha,
    )

    # Save
    import pickle
    with open(output_dir / "r2_dict.pkl", "wb") as f:
        pickle.dump(r2_dict, f)
    with open(output_dir / "mse_dict.pkl", "wb") as f:
        pickle.dump(mse_dict, f)
    with open(output_dir / "probe_weights.pkl", "wb") as f:
        pickle.dump(probe_weights, f)

    print(f"Saved to {output_dir}")
    print(f"Best R²: {max(r2_dict.values()):.4f} at {max(r2_dict, key=r2_dict.get)}")
    return r2_dict, mse_dict, probe_weights
