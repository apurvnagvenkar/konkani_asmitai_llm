from transformers import AutoTokenizer
from datasets import load_dataset
import argparse

def count_total_tokens(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("anag007/asmitai_wiki_konkani_dataset", split="train")
    texts = dataset["text"]

    total_tokens = 0
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)

    print(f"Total tokens in dataset for model '{model_name}': {total_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    args = parser.parse_args()

    count_total_tokens(args.model)