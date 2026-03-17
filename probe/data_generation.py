from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Gemma3ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import is_correct

# Be sure to authenticate with Hugging Face Hub from the terminal so you can load the model and Truthful QA dataset
def format_prompt(question):
    """Puts the question from Truthful QA dataset into prompt format so Gemma response is standardized."""
    chat = [
        {
            "role": "user",
            "content": (
                "Answer the following question concisely. "
                "Do not ask additional questions. \n"
                f"Question: {question}"
            )
        }
    ]
    return processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def sample_model(prompt, temperature=0.7, max_new_tokens=100):
    """Uses a prompt to query the Gemma model and get a response."""
    inputs = processor.tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.3,
            #pad_token_id=processor.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    response = processor.decode(output[0][prompt_len:], skip_special_tokens=True)
    return response.replace("<end_of_turn>", "").strip()

def get_answer_confidence(prompt, answer):
    """Measures the model's own probability assigned to its answer."""
    inputs = processor.tokenizer(prompt + answer, return_tensors="pt").to(model.device)
    prompt_len = processor.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits[:, prompt_len-1:-1]  # answer tokens only
        probs = torch.softmax(logits, dim=-1) # 1 x answer token length x vocab size
        # print(f'probs: {probs}')
        # entropy over answer tokens — low entropy = high certainty
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
        # mean prob of chosen tokens
        chosen_ids = inputs["input_ids"][:, prompt_len:]
        # print(f'chosen ids: {chosen_ids}')
        chosen_probs = probs[0].gather(1, chosen_ids.T).squeeze() # get probs for actual token selected at each position
        mean_prob = chosen_probs.mean().item()

    return {'entropy': entropy, 'mean_token_prob': mean_prob}


def main():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    dataset = load_dataset("truthful_qa", "generation")["validation"]
    model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it", device_map="cuda:0")
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    
    # Train test split of Truthful QA
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_data = dataset.select(train_idx)
    test_data  = dataset.select(test_idx)
    
    # Generate responses for each question in the training dataset
    questions = train_data['question']
    rows = []
    for i, question in enumerate(questions):
        prompt = format_prompt(question)
        answer = sample_model(prompt)
        dict = get_answer_confidence(prompt, answer)
        correct = is_correct(answer, train_data[i]['correct_answers'], train_data[i]['incorrect_answers'])
        # acc = empirical_accuracy_batched(
        #     embedder, responses,
        #     correct_embs_all[i], incorrect_embs_all[i]
        # )
        rows.append({'question': question, 'response': answer, 'token entropy': dict['entropy'], 'mean token prob': dict['mean_token_prob'], 'correct': correct})
        if i % 10 == 0:
            print(f"[{i}/{len(questions)}] done")

    # Save question, Gemma response, token entropy and probabilities, and whether the response is correct or not to a dataframe
    final_ds = pd.DataFrame(rows)
    final_ds.to_csv("train_responses.csv")
    
    # Save test data split
    test_data.to_pandas().to_csv("test_data.csv")
    

if __name__ == "__main__":
    main()
    
    

