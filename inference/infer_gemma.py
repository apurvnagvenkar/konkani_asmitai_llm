from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# -------------------------
# Configuration
# -------------------------
model_id = "google/gemma-3-1b-it"     # base model name or path
peft_id = None                          # set to PEFT adapter path/name if using

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(base_id, peft_adapter=None):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)

    # Load base causal LM
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto" if device.startswith("cuda") else None,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True
    )

    # Wrap with PEFT adapter if provided
    if peft_adapter:
        model = PeftModel.from_pretrained(model, peft_adapter, device_map="auto")

    model.to(device)
    model.eval()
    return tokenizer, model

# Load tokenizer and model (with or without PEFT)
tokenizer, model = load_model(model_id, peft_id)

# -------------------------
# Chat messages
# -------------------------
messages = [
    [
        {
            "role": "user",
            "content": [{"type": "text", "text": "दिल्ल्या दोंगर माळेंतल्या सगळ्यांत ऊंच तेमकाचें नांव सांगात.\nरॉकी पर्वत"}]
        },
    ],
]

# -------------------------
# Tokenize & Generate
# -------------------------
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device).to(torch.bfloat16)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=64)

# -------------------------
# Decode & Print
# -------------------------
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(decoded)
