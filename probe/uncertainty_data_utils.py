"""
uncertainty_data_utils.py

Builds (chats, labels) for Step 1 of the uncertainty probe pipeline.

Goal: train a probe that predicts "will the model get this right?"
from the internal activations at each attention head.

If a head's activations reliably predict correctness, that head
is encoding the model's internal certainty about the answer.

Labels
------
  "correct"   (→ 1 after LabelEncoder)  model's answer matches a correct answer
  "incorrect" (→ 0 after LabelEncoder)  model's answer does not match

No second model needed. No assertiveness scorer. No entropy.
Ground truth comes directly from TruthfulQA's correct_answers field.

The chats passed to extract_activation include the model's answer as the
assistant turn — same format as the sycophancy pipeline — because we want
to capture the activation state at the moment the model has committed to
an answer, not just at the question.
"""

import torch
from tqdm.auto import tqdm
import pandas as pd


def get_model_answer(
    question_chat: list, model, processor, max_new_tokens: int = 80
) -> str:
    """
    Generates a short answer from the model for one question.

    question_chat: [{"role": "user", "content": "...question..."}]
    Returns the decoded answer string.
    """
    is_gemma = "gemma" in str(type(processor)).lower()

    templated = processor.apply_chat_template(
        question_chat, add_generation_prompt=True, tokenize=False
    )

    if is_gemma:
        input_ids = processor(text=templated, return_tensors="pt")["input_ids"].to(
            model.device
        )
    else:
        input_ids = processor(templated, return_tensors="pt")["input_ids"].to(
            model.device
        )

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=(
                processor.tokenizer.eos_token_id if is_gemma else processor.eos_token_id
            ),
        )

    new_tokens = generated[0][input_ids.shape[1] :]
    if is_gemma:
        answer = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        answer = processor.decode(new_tokens, skip_special_tokens=True)

    return answer.strip()


def is_correct(model_answer: str, correct_answers: list[str]) -> bool:
    """
    Checks whether the model's answer matches any of the correct answers.

    Uses simple case-insensitive substring matching — good enough for
    TruthfulQA where correct answers are short factual strings.
    You can make this stricter (exact match) or looser (word overlap)
    depending on how many examples survive.
    """
    model_lower = model_answer.lower()
    for correct in correct_answers:
        if correct.lower() in model_lower or model_lower in correct.lower():
            return True
    return False


def construct_correctness_data(
    ds_split,
    model,
    processor,
    max_new_tokens: int = 80,
) -> tuple[list, list]:
    """
    Runs the model on each TruthfulQA question, checks if the answer
    is correct, and returns (chats, labels) ready for train.py.

    Parameters
    ----------
    ds_split : HuggingFace dataset split
        The TruthfulQA training split — same one train.py already uses:
            ds = load_dataset("truthfulqa/truthful_qa", "generation")
            split_ds = ds["validation"].train_test_split(test_size=0.2, seed=3407)
            ds_train = split_ds["train"]   ← pass this

    model, processor : your main Gemma/Qwen model
        Already loaded in train.py — just pass them through.

    Returns
    -------
    chats  : list of chat dicts
        Each entry is the full exchange:
            [{"role": "user", "content": question},
             {"role": "assistant", "content": model_answer}]
        This is what gets tokenized and passed to extract_activation.

    labels : list of str
        "correct" or "incorrect" per example.
        LabelEncoder in train.py converts these to 0/1 automatically.
    """
    chats = []
    labels = []

    n_correct = 0
    n_incorrect = 0

    for row in tqdm(ds_split, desc="Building correctness dataset"):
        question = row["question"]
        correct_answers = row["correct_answers"]  # list of strings

        question_chat = [{"role": "user", "content": question}]

        # generate the model's actual answer
        answer = get_model_answer(question_chat, model, processor, max_new_tokens)

        # label from ground truth
        correct = is_correct(answer, correct_answers)

        full_chat = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        chats.append(full_chat)

        if correct:
            labels.append("correct")
            n_correct += 1
        else:
            labels.append("incorrect")
            n_incorrect += 1

    print(f"\nCorrect: {n_correct}, Incorrect: {n_incorrect}")
    print(f"Class balance: {n_correct / max(len(labels), 1):.1%} correct")

    if n_correct == 0 or n_incorrect == 0:
        print("WARNING: one class is empty. Check is_correct() matching logic.")

    return chats, labels


"""Builds (chats, labels) for Step 2 — the suppression probe.
 
Labels:
    1 = "miscalibrated"  — model is uncertain (low sampling accuracy)
                           BUT answer sounds confident
    0 = "calibrated"     — model is uncertain (low sampling accuracy)
                           AND answer sounds hedged
 
Your partner provides a CSV with columns:
    question, accuracy, answer, assertiveness_score
 
Where:
    accuracy          = fraction of N samples that were correct (0.0–1.0)
    answer            = the modal/representative answer text
    assertiveness_score = score from the epistemic integrity scorer (0–10)
 
We keep only rows where accuracy < uncertainty_threshold
(the model is genuinely uncertain — these are the interesting cases)
then split on assertiveness_score.
 
The chat format matches probe_data_utils.py exactly — same to_message()
structure so train.py's tokenization works identically.
"""


def construct_suppression_data(
    scored_csv_path: str,
    uncertainty_threshold: float = 0.5,
    assertiveness_threshold: float = 5.0,
) -> tuple[list, list]:
    """
    Parameters
    ----------
    scored_csv_path : str
        Path to the CSV your partner produces.
        Required columns: question, accuracy, answer, assertiveness_score

    uncertainty_threshold : float
        Keep only rows where accuracy < this value.
        0.5 means "model gets it right less than half the time" = uncertain.
        Adjust based on your data distribution.

    assertiveness_threshold : float
        Split uncertain examples:
            assertiveness_score >= threshold → label 1 (miscalibrated)
            assertiveness_score <  threshold → label 0 (calibrated)

    Returns
    -------
    chats  : list of chat dicts  — same format as construct_data() returns
    labels : list of int         — 1 = miscalibrated, 0 = calibrated
    """
    df = pd.read_csv(scored_csv_path)

    required = {"question", "accuracy", "answer", "assertiveness_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # keep only uncertain examples
    uncertain = df[df["accuracy"] < uncertainty_threshold].copy()
    print(f"Total examples: {len(df)}")
    print(f"Uncertain (accuracy < {uncertainty_threshold}): {len(uncertain)}")
    print(f"Dropped (model was confident): {len(df) - len(uncertain)}")

    chats = []
    labels = []

    for _, row in uncertain.iterrows():
        question = row["question"]
        answer = row["answer"]

        # build chat in the same format as probe_data_utils.to_message()
        # using the 'truthful' template since it's the simplest —
        # just question + answer, no challenge turn needed
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        chats.append(chat)

        if row["assertiveness_score"] >= assertiveness_threshold:
            labels.append(1)  # miscalibrated — uncertain inside, confident outside
        else:
            labels.append(0)  # calibrated    — uncertain inside, hedged outside

    n_misc = sum(labels)
    n_cal = len(labels) - n_misc
    print(f"\nMiscalibrated (label 1): {n_misc}")
    print(f"Calibrated    (label 0): {n_cal}")
    print(f"Class balance: {n_misc/max(len(labels),1):.1%} miscalibrated")

    if n_misc == 0 or n_cal == 0:
        print("WARNING: one class is empty.")
        print("  If all miscalibrated: lower assertiveness_threshold")
        print("  If all calibrated:    raise assertiveness_threshold")
        print("  If too few examples:  raise uncertainty_threshold")

    return chats, labels
