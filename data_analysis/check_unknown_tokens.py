# script_1_unknown_tokens.py

from transformers import AutoTokenizer
from datasets import load_dataset
import argparse

def get_unknown_tokens(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("anag007/asmitai_wiki_konkani_dataset", split="train")
    texts = dataset["text"]

    unk_token_id = tokenizer.unk_token_id
    if unk_token_id is None:
        print(f"Tokenizer for {model_name} does not use an <unk> token.")
        return

    for idx, text in enumerate(texts):
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        unknown_tokens = [tok for tok, tid in zip(tokens, token_ids) if tid == unk_token_id]

        if unknown_tokens:
            print(f"Sample {idx} | Unknown Tokens: {unknown_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    args = parser.parse_args()

    get_unknown_tokens(args.model)
