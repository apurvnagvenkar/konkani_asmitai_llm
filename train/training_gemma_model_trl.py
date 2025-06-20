import argparse
import yaml
import os
import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from trl import SFTTrainer, SFTConfig
import wandb

# -------------------------
# Parse command-line arguments
# -------------------------
parser = argparse.ArgumentParser(description="Fine-tune LLM on multiple GPUs")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
args = parser.parse_args()

# Load YAML config
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
config = load_config(args.config)

# Paths setup
model_name = config['model_name']
dataset_name = config['dataset_name']
exp_dir = os.path.join(config['experiment_parent_path'], config['experiment_name'])
output_dir = os.path.join(exp_dir, 'output')
logging_dir = os.path.join(exp_dir, 'logs')

# Accelerator for multi-GPU
accelerator = Accelerator()

# Tokenizer and chat template
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# BitsAndBytes 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load base model (NO device_map!)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

# Apply PEFT LoRA
lora_cfg = LoraConfig(
    task_type="CAUSAL_LM",
    r=config['lora']['r'],
    lora_alpha=config['lora']['lora_alpha'],
    target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ],
    lora_dropout=config['lora']['lora_dropout'],
    bias=config['lora']['bias'],
)
model = get_peft_model(model, lora_cfg)
model = accelerator.prepare(model)

# Load and prepare datasets
dataset = load_dataset(dataset_name)
if "validation" in dataset:
    train_ds = dataset['train']
    valid_ds = dataset['validation']
else:
    full_ds = dataset['train']
    # Use only 5000 for training and up to 5000 for validation (if enough available)
    val_size = 5000 #min(5000, int(0.1 * len(full_ds)))
    split = full_ds.train_test_split(
        test_size=val_size,
        seed=42
    )
    train_ds = split["train"]
    valid_ds = split["test"]

# Formatting with apply_chat_template
def format_fn(examples):
    texts = []
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        final_instruction = ""
        if input.strip() == "nan":
            final_instruction = instruction
        else:
            final_instruction = "%s\n %s" % (instruction, input)
        message = [
            {"role": "user", "content": "%s" % final_instruction},
            {"role": "assistant", "content": "%s" % output}
        ]

        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {'text': texts}

train_ds = train_ds.map(format_fn, batched=True)
valid_ds = valid_ds.map(format_fn, batched=True)

# Preprocess: mask prompt labels
IN_TAG = '<start_of_turn>user\n'
OUT_TAG = '<start_of_turn>model\n'

def preprocess_fn(examples):
    input_ids_list, labels_list = [], []
    for text in examples['text']:
        if OUT_TAG in text:
            prompt_part, response = text.split(OUT_TAG, 1)
            prompt_text = prompt_part + OUT_TAG
        else:
            prompt_text, response = text, ''
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        resp_ids = tokenizer(response, add_special_tokens=False).input_ids
        eos = tokenizer.eos_token_id
        input_ids = prompt_ids + resp_ids + [eos]
        labels = [-100] * len(prompt_ids) + resp_ids + [eos]
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    return {'input_ids': input_ids_list, 'labels': labels_list}

REMOVE_COLS = [
    'instruction', 'input', 'output'
]
train_ds = train_ds.map(
    preprocess_fn,
    batched=True,
    remove_columns=[c for c in REMOVE_COLS if c in train_ds.column_names]
)
valid_ds = valid_ds.map(
    preprocess_fn,
    batched=True,
    remove_columns=[c for c in REMOVE_COLS if c in valid_ds.column_names]
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Setup TRL SFTTrainer config
sft_args = SFTConfig(
    per_device_train_batch_size=config['sft_config']['per_device_train_batch_size'],
    gradient_accumulation_steps=config['sft_config']['gradient_accumulation_steps'],
    warmup_steps=config['sft_config']['warmup_steps'],
    save_steps=config['sft_config']['save_steps'],
    eval_steps=config['sft_config']['eval_steps'],
    eval_strategy="steps",
    prediction_loss_only=True,
    logging_steps=config['sft_config']['logging_steps'],
    num_train_epochs=config['sft_config']['num_train_epochs'],
    learning_rate=config['sft_config']['learning_rate'],
    optim=config['sft_config']['optim'],
    weight_decay=config['sft_config']['weight_decay'],
    lr_scheduler_type=config['sft_config']['lr_scheduler_type'],
    seed=config['sft_config']['seed'],
    report_to=config['sft_config']['report_to'],
    output_dir=output_dir,
    logging_dir=logging_dir,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=data_collator,
    args=sft_args,
)

# WandB init on main
if accelerator.is_main_process and 'wandb' in sft_args.report_to:
    wandb.init(
        project=config.get('wandb', {}).get('project', 'konkani-finetune'),
        name=config['experiment_name'],
        entity=config.get('wandb', {}).get('entity'),
        config=config,
        notes=config.get("notes", "")
    )

# Train
stats = trainer.train()
if accelerator.is_main_process:
    print(stats)

# CLI to run:
# accelerate launch --num_processes=<NUM_GPUS> --mixed_precision=fp16 script.py --config config.yaml
