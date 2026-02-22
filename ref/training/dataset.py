from pathlib import Path
import json
from config import TRANSLATION_PROMPT

def load_training_data(data_path, max_samples = None):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    with open(data_path, "r", encoding = "utf-8") as f:
        data = json.load(f)
    
    if max_samples is not None:
        data = data[ : max_samples]

    return data

def prepare_training_data(raw_data, prompt_template = TRANSLATION_PROMPT):
    training_data = []

    for item in raw_data:
        input_text = item.get("input", "").strip()
        output_text = item.get("output", "").strip()
        prompt = prompt_template.format(input = input_text, output = output_text)
        training_data.append({"text": prompt})

    return training_data

def tokenize_training_data(training_data, tokenizer):
    tokenized_data = []

    for item in training_data:
        text = item["text"]
        tokenized_item = tokenizer(
            text,
            truncation = True,
            padding = False,
            max_length = tokenizer.model_max_length,
            return_tensors = None
        )

        tokenized_data.append(tokenized_item)

    return tokenized_data

def tokenize_sample(sample, tokenizer):
    encoded = tokenizer(
        sample["text"],
        truncation = True,
        padding = False,
        max_length = tokenizer.model_max_length
    )

    encoded["labels"] = encoded["input_ids"].copy()
    return encoded