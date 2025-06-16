import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, Gemma3ForCausalLM
from tqdm import tqdm
import argparse

# -------------------------
# Parse command-line arguments
# -------------------------
parser = argparse.ArgumentParser(description="Generate model outputs from chat-style instructions")
parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset path")
parser.add_argument("--model", type=str, required=True, help="Model name or path")
parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path")
args = parser.parse_args()

# -------------------------
# Load tokenizer and model
# -------------------------
model_id = args.model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Gemma3ForCausalLM.from_pretrained(model_id).eval().to("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load dataset
# -------------------------
dataset = load_dataset(args.dataset)
test_data = dataset["test"]

# -------------------------
# Process dataset
# -------------------------
records = []
device = model.device

for example in tqdm(test_data, desc="Generating responses"):
    instruction = example["instruction"]
    input_text = example["input"]
    ground_truth_output = example["output"]
    if input_text.strip() == "nan":
        final_instruction = instruction
    else:
        final_instruction = "%s\n %s" % (instruction, input_text)

    # Create chat format
    message = [
        {
            "role": "user",
            "content": [{"type": "text", "text": final_instruction}],
        }
    ]

    # Tokenize using Gemma chat template
    inputs = tokenizer.apply_chat_template(
        [message],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device).to(torch.bfloat16)
    # Generate output
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=64)

    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    records.append({
        "instruction": instruction,
        "input": input_text,
        "output": ground_truth_output,
        "model_output": decoded_output,
    })

# -------------------------
# Save to CSV
# -------------------------
df = pd.DataFrame(records)
df.to_csv(args.output_csv, index=False)
print(f"Saved output to {args.output_csv}")
