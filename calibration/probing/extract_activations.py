"""
Extract MHA activations at the last token for assertion texts.

Uses forward hooks on attention head outputs (same as probe/extract_activation.py).
Activations are extracted at the last token position of each text.
"""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add probe directory for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROBE_DIR = _REPO_ROOT / "probe"
if str(_PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBE_DIR))

from utils import load_model
import extract_activation


def _format_and_tokenize(
    text: str,
    processor,
    model_id: str,
    max_length: int = 2048,
    device: str = "cuda",
) -> torch.Tensor:
    """Format text as chat and tokenize. Returns input_ids tensor."""
    if "gemma" in model_id.lower():
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ]

    inputs_str = processor.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    if "gemma" in model_id.lower():
        encoded = processor.tokenizer(
            text=inputs_str,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    else:
        encoded = processor(
            inputs_str,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    return encoded["input_ids"].squeeze(0).to(device)


def extract_activations_batch(
    model,
    processor,
    texts: list[str],
    model_id: str,
    max_length: int = 2048,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract MHA activations at last token for each text.

    Args:
        model: Frozen LLM.
        processor: Tokenizer/processor.
        texts: List of assertion texts.
        model_id: Model identifier (e.g. 'gemma-3', 'llama-3.2').
        max_length: Max token length (truncate longer texts).
        device: Device for model.

    Returns:
        Tensor of shape (n_texts, n_layers, n_heads, head_dim).
    """
    model.eval()
    activations_list = []
    for text in tqdm(texts, desc="Extracting activations"):
        input_ids = _format_and_tokenize(text, processor, model_id, max_length, device)
        act = extract_activation.extract_mha_activation(model, processor, input_ids)
        activations_list.append(act.cpu())
    return torch.stack(activations_list)
