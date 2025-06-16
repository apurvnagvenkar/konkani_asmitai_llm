from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer
import argparse
import yaml

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fine-tune Konkani LLM")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
args = parser.parse_args()

# Load YAML config
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

model_name = config["model_name"]
dataset_name = config["dataset_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)

experiment_parent_path = config["experiment_parent_path"]
experiment_name = config["experiment_name"]

output_dir = experiment_parent_path + "/" + experiment_name +"/output"
logging_dir = experiment_parent_path + "/" + experiment_name +"/logs"

model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=config["train"]["max_seq_length"],
    load_in_4bit=config["train"]["load_in_4bit"],
    load_in_8bit=config["train"]["load_in_8bit"],
    full_finetuning=config["train"]["full_finetuning"],
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    lora_dropout=config["lora"]["lora_dropout"],
    bias=config["lora"]["bias"],
    random_state=config["lora"]["random_state"],
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

dataset = load_dataset(dataset_name)
train_dataset = dataset["test"]
valid_dataset = dataset["validation"]


train_dataset = standardize_data_formats(train_dataset)
valid_dataset = standardize_data_formats(valid_dataset)


def formatting_prompts_func(examples):
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
       {"role": "assistant", "content": "%s" % output }
        ]

       text = tokenizer.apply_chat_template(message, tokenize = False)
       texts.append(text)
   return { "text" : texts, }

train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
valid_dataset = valid_dataset.map(formatting_prompts_func, batched = True)

print(train_dataset[100]["text"])


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=config["sft_config"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["sft_config"]["gradient_accumulation_steps"],
        warmup_steps=config["sft_config"]["warmup_steps"],
        save_steps=config["sft_config"]["save_steps"],
        eval_steps=config["sft_config"]["eval_steps"],
        logging_steps=config["sft_config"]["logging_steps"],
        num_train_epochs=config["sft_config"]["num_train_epochs"],
        learning_rate=config["sft_config"]["learning_rate"],
        optim=config["sft_config"]["optim"],
        weight_decay=config["sft_config"]["weight_decay"],
        lr_scheduler_type=config["sft_config"]["lr_scheduler_type"],
        seed=config["sft_config"]["seed"],
        report_to=config["sft_config"]["report_to"],
        dataset_num_proc=config["sft_config"]["dataset_num_proc"],
        output_dir=output_dir,
        logging_dir=logging_dir,
        metric_fo_best_model="eval_loss"
    ),
)


trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)



# print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
#
# tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")


trainer_stats = trainer.train()
print(trainer_stats)