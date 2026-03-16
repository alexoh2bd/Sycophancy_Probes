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
