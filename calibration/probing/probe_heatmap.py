"""
Visualize R² (or MSE) across (layer, head) as a heatmap.

Analogous to Figure 3 in the sycophancy paper.
"""

import pickle
from pathlib import Path


def plot_r2_heatmap(
    r2_dict: dict,
    num_layers: int,
    num_heads: int,
    output_path: Path | str | None = None,
):
    """
    Plot heatmap of R² per (layer, head).

    Args:
        r2_dict: Mapping "layer_head" -> R² value.
        num_layers, num_heads: Model config.
        output_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.zeros((num_layers, num_heads))
    for key, val in r2_dict.items():
        layer, head = map(int, key.split("_"))
        matrix[layer, head] = val

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("R² per (layer, head) - Assertiveness Probe")
    plt.colorbar(im, ax=ax, label="R²")
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")
    plt.close()


def load_and_plot(results_dir: Path | str, model_id: str = "gemma-3"):
    """Load R² from pickle and plot. Infer num_layers/num_heads from keys."""
    results_dir = Path(results_dir)
    with open(results_dir / "r2_dict.pkl", "rb") as f:
        r2_dict = pickle.load(f)
    keys = list(r2_dict.keys())
    layers = [int(k.split("_")[0]) for k in keys]
    heads = [int(k.split("_")[1]) for k in keys]
    num_layers = max(layers) + 1
    num_heads = max(heads) + 1
    plot_r2_heatmap(r2_dict, num_layers, num_heads, results_dir / "r2_heatmap.png")
