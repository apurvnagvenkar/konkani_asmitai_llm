from data_preparation.lib.prepare_noisified_data import augment_konkani_dataset
from data_preparation.lib.utils import create_dataset_dict, upload_to_huggingface

tokenizer_name = "google/gemma-3-12b-it"

repo_name = "asmitai_konkani_gemma-3-12b_noisified_alpaca_instruction_data"
train_df = augment_konkani_dataset(upsample=10,  tokenizer_name=tokenizer_name, instruction_data=True, type="train")
validation_df = augment_konkani_dataset(upsample=1,  tokenizer_name=tokenizer_name, instruction_data=True, type="validation")
test_df = augment_konkani_dataset(upsample=1,  tokenizer_name=tokenizer_name, instruction_data=True, type="test")

dataset_dict= create_dataset_dict(train_df, validation_df, test_df)
upload_to_huggingface(dataset_dict, repo_name)