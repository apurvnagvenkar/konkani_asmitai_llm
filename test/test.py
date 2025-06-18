import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import argparse

# -------------------------
# Parse command-line arguments
# -------------------------
parser = argparse.ArgumentParser(description="Generate model outputs from chat-style instructions with optional PEFT")
parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset path")
parser.add_argument("--model", type=str, required=True, help="Base model name or path")
parser.add_argument("--peft_model", type=str, default=None, help="Optional PEFT model name or path")
parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path")
args = parser.parse_args()

# -------------------------
# Load tokenizer and model
# -------------------------
model_id = args.model

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
# Load base causal LM
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"":0},#"auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# If a PEFT model is provided, wrap the base model
if args.peft_model:
    peft_id = args.peft_model
    model = PeftModel.from_pretrained(model, peft_id, device_map="auto")

model.eval()
device = next(model.parameters()).device

# -------------------------
# Load dataset
# -------------------------
dataset = load_dataset(args.dataset)
test_data = dataset.get("test", dataset[list(dataset.keys())[0]])

# -------------------------
# Process dataset
# -------------------------
records = []
c =0 
for example in tqdm(test_data, desc="Generating responses"):
    if c == 100:
        break
    c+=1
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    ground_truth_output = example.get("output", "")

    # Combine instruction and input
    if input_text and input_text.strip().lower() != "nan":
        prompt = f"{instruction}\n{input_text}"
    else:
        prompt = instruction

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=False
    ).to(device)

    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]  # number of tokens in the prompt

    # Generate output
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
    generated_ids = outputs[0][prompt_len:]
    model_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(model_generated_text)

    records.append({
        "instruction": instruction,
        "input": input_text,
        "output": ground_truth_output,
        "model_output": model_generated_text,
    })

# -------------------------
# Save to CSV
# -------------------------

df = pd.DataFrame(records)
df.to_csv(args.output_csv, index=False)
print(f"Saved output to {args.output_csv}")
