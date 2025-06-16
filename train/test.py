from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer

model_name = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
dataset_name = "anag007/asmitai_konkani_gemma-3-12b_noisified_alpaca_instruction_data"
tokenizer = AutoTokenizer.from_pretrained(model_name)




dataset = load_dataset(dataset_name)
train_dataset = dataset["test"]
valid_dataset = dataset["validation"]




def formatting_prompts_func(examples):
   print(examples)
   instruction = examples["instruction"]
   input = examples["input"]
   output = examples["output"]
   messages = [
       {"role": "system", "content": "You are a friendly chatbot who always responds in Konkani" },
       {"role": "user", "content": "%s\n %s" % (instruction, input)},
       {"role": "assistant", "content": "%s" % output }
   ]

   texts = [tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
valid_dataset = valid_dataset.map(formatting_prompts_func, batched = True)

print(train_dataset[100]["text"])

