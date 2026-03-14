# Sycophancy Detection in Large Language Models

This project provides tools for training linear probes to detect sycophancy and truthfulness in large language models by analyzing internal activations across different model components. It also includes an **assertiveness probe pipeline** for epistemic calibration using the epistemic-integrity dataset.

## Overview

The project trains linear classifiers (probes) on activations extracted from different components of LLMs:
- **Multi-Head Attention (MHA)** outputs
- **MLP layer** outputs  
- **Residual stream** activations

**Sycophancy probes** are trained on the TruthfulQA dataset. **Assertiveness probes** use ridge regression on the epistemic-integrity dataset (`train_data.csv` / `test_data.csv`) to predict continuous assertiveness scores and enable steering.

## Features

- **Activation Extraction**: Extract activations from MHA heads, MLP layers, and residual streams
- **Linear Probe Training**: Train binary classifiers (sycophancy) or ridge regression probes (assertiveness) on extracted activations
- **Steering**: Apply interventions (h - α × direction) to steer model behavior at inference
- **Visualization**: Generate R² heatmaps across layers/heads
- **Batch Processing**: Support for SLURM job submission
- **Multiple Models**: Compatible with Gemma and Llama model families

## Project Structure

```
.
├── probe/
│   ├── train.py                    # Sycophancy: linear probe training
│   ├── train_epint.py              # Assertiveness: ridge regression probe training
│   ├── extract_activation.py       # Activation extraction utilities
│   ├── compute_proj_std.py         # Projection std for scale × std × direction
│   ├── inference_epint.py          # Assertiveness steering inference
│   ├── evaluate_epint.py          # Score assertiveness of model outputs
│   ├── plot_probe_heatmap.py       # R² heatmap visualization
│   └── utils.py                    # load_model, load_ep_data, etc.
├── inference_mha.py                # Sycophancy: MHA probe inference
├── inference_mlp.py                # MLP probe inference
├── inference_residual.py           # Residual probe inference
├── run_train_inference.sh          # SLURM: sycophancy pipeline
├── run_train_inference_epint.sh   # SLURM: assertiveness pipeline
└── pyproject.toml                  # uv / pip dependencies
```

## Setup

### Prerequisites

- Python 3.14+
- CUDA-compatible GPU (recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
cd /path/to/sycophancy
uv sync
```

## UV Commands

All commands use `uv run python` to ensure the correct environment. Run from the project root.

### Sycophancy Pipeline (TruthfulQA)

```bash
# 1. Train linear probes
uv run python probe/train.py --model_id gemma-3 --activation_type mha --concept sycophancy --probe_type linear

# 2. Compute projection std (for scale × std × direction)
uv run python probe/compute_proj_std.py --model_id gemma-3 --activation_type mha --concept sycophancy --probe_type linear

# 3. Run inference with steering
uv run python inference_mha.py --model_id gemma-3 --dataset_id truthfulqa --concept sycophancy --probe_type linear
```

### Assertiveness Pipeline (epistemic-integrity)

```bash
# 1. Train ridge regression probes (90/10 train/val, test on test_data.csv)
uv run python probe/train_epint.py --model_id gemma-3 --wandb

# 2. Compute projection std (optional; for scale × std × direction)
uv run python probe/compute_proj_std.py --model_id gemma-3 --activation_type mha --concept assertiveness

# 3. Run steering inference (scale < 0 steers down assertiveness)
uv run python probe/inference_epint.py --model_id gemma-3 --k_heads 8 --scale 2.5 --wandb

# 4. Plot R² heatmap
uv run python probe/plot_probe_heatmap.py --model_id gemma-3 --wandb

# 5. Evaluate assertiveness of outputs (requires inference CSV)
uv run python probe/evaluate_epint.py --csv predictions_assertiveness/epint_gemma-3_k8_scale2.5.csv --wandb
```

### SLURM Batch Jobs

```bash
# Sycophancy pipeline
sbatch run_train_inference.sh

# Assertiveness pipeline
sbatch run_train_inference_epint.sh
```

## Output

### Sycophancy Probes

1. **Trained Probes**: Saved as `.pth` files in `probe/trained_probe_{concept}/{model_name}/`
   - MHA: `linear_probe_{layer}_{head}.pth`
   - MLP: `linear_probe_mlp_{layer}.pth`
   - Residual: `linear_probe_residual_{layer}.pth`

2. **Accuracy Dictionary**: `{probe_type}_accuracies_dict_{activation_type}.pkl`

3. **Projection Std**: `{probe_type}_std_mha_{layer}_{head}.pt` (for scale × std × direction)

### Assertiveness Probes

1. **Probe Weights**: `probe/trained_probe_assertiveness/{model_name}/`
   - `probe_weights.pkl` — ridge coef, intercept, scaler per (layer, head)
   - `r2_dict.pkl`, `mse_dict.pkl`, `test_r2_dict.pkl`, `test_mse_dict.pkl`
   - `linear_accuracies_dict_mha.pkl` (R² for top-k selection)

2. **Inference Output**: `predictions_assertiveness/epint_{model}_k{k}_scale{scale}.csv`

3. **Visualizations**: R² heatmap across layers/heads

## Datasets

- **Sycophancy**: [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) — training/validation split 80/20
- **Assertiveness**: `epistemic-integrity/scibert-finetuning/data/` — `train_data.csv` (90/10 split), `test_data.csv` (held-out)

## Model Support

Currently tested with:
- Google Gemma models (gemma-2, gemma-3)
- Meta Llama models (Llama-3.2)

The code is designed to be extensible to other transformer-based language models.

## Notebooks

Interactive Jupyter notebooks are provided for:
- `attention_map.ipynb`: Visualizing attention patterns
- `compute_metrics.ipynb`: Computing evaluation metrics
- `get_prediction.ipynb`: Analyzing model predictions
- `plots.ipynb`: Creating custom visualizations

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## Contact

[Add contact information here]
