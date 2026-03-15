"""
Evaluate inference_epint outputs: score assertiveness of initial vs steered answers, report to wandb.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
sys.path.insert(0, current_dir)

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import wandb


def load_assertiveness_model(model_name="pedropei/sentence-level-certainty", checkpoint_path=None, device="cuda"):
    """Load assertiveness scorer (base or fine-tuned)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def score_texts(model, tokenizer, texts, device="cuda", batch_size=32, max_length=512):
    """Score assertiveness for a list of texts. Returns array of scores."""
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Filter empty
        batch = [t if t and str(t).strip() else " " for t in batch]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        preds = out.logits.squeeze(-1).cpu().numpy()
        if preds.ndim == 0:
            preds = np.array([float(preds)])
        scores.extend(preds.tolist())
    return np.array(scores, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference_epint outputs with assertiveness scorer.")
    parser.add_argument("--csv", type=str, required=True, help="Path to inference CSV (from inference_epint.py)")
    parser.add_argument("--scorer_model", type=str, default="pedropei/sentence-level-certainty")
    parser.add_argument("--scorer_checkpoint", type=str, default=None, help="Optional fine-tuned checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    if "initial_answer" not in df.columns or "steered_answer" not in df.columns:
        raise ValueError(f"CSV must have 'initial_answer' and 'steered_answer' columns. Got: {list(df.columns)}")

    initial_texts = df["initial_answer"].fillna("").astype(str).tolist()
    steered_texts = df["steered_answer"].fillna("").astype(str).tolist()

    print(f"Loading assertiveness scorer: {args.scorer_model}")
    model, tokenizer = load_assertiveness_model(
        args.scorer_model, args.scorer_checkpoint, args.device
    )

    print("Scoring initial answers...")
    initial_scores = score_texts(model, tokenizer, initial_texts, args.device)
    print("Scoring steered answers...")
    steered_scores = score_texts(model, tokenizer, steered_texts, args.device)

    mean_initial = float(np.mean(initial_scores))
    mean_steered = float(np.mean(steered_scores))
    mean_change = mean_steered - mean_initial
    std_initial = float(np.std(initial_scores))
    std_steered = float(np.std(steered_scores))

    print(f"Mean assertiveness (initial): {mean_initial:.4f} ± {std_initial:.4f}")
    print(f"Mean assertiveness (steered): {mean_steered:.4f} ± {std_steered:.4f}")
    print(f"Mean change (steered - initial): {mean_change:.4f}")

    if args.wandb:
        wandb.init(
            project="assertiveness-steering-eval",
            config={
                "csv": args.csv,
                "scorer_model": args.scorer_model,
                "n_samples": len(df),
            },
            name=os.path.basename(args.csv).replace(".csv", ""),
        )
        wandb.log({
            "eval/mean_assertiveness_initial": mean_initial,
            "eval/mean_assertiveness_steered": mean_steered,
            "eval/mean_change": mean_change,
            "eval/std_initial": std_initial,
            "eval/std_steered": std_steered,
        })
        # Log per-sample change distribution
        changes = steered_scores - initial_scores
        wandb.log({
            "eval/mean_per_sample_change": float(np.mean(changes)),
            "eval/pct_decreased": float(np.mean(changes < 0) * 100),
        })
        wandb.finish()

    return mean_initial, mean_steered, mean_change


if __name__ == "__main__":
    main()
