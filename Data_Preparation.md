# Konkani Instruction-Tuning Dataset Preparation with Gemma Tokenizer

## Objective

* Prepare high-quality instruction-tuning datasets for the Konkani language.
* Convert raw monolingual text into synthetic instruction–source–target triples using noise functions.
* Evaluate tokenizer efficiency to select the best model family for downstream fine-tuning (Gemma vs. Llama).

---

## Data Overview

| Data Source                                                    | Size           | Scripts                   | Notes                                          |
| -------------------------------------------------------------- | -------------- | ------------------------- | ---------------------------------------------- |
| **Instruction data** – saillab/alpaca-konkani-cleaned          | ≈ 29k examples | Devanagari                | MT-translated Alpaca-style; single-script only |
| **Monolingual data** – anag007/asmitai\_wiki\_konkani\_dataset | 2,992 articles | Devanagari, Kannada, Romi | Raw Wikipedia text; multi-script support       |

---

## Problem Statements

### 🪤 Monolingual fine-tuning:

* Can an instruction-tuned model retain its instruction-following capability when fine-tuned solely on unlabeled Konkani text?

### 🌐 Cross-script generalization:

* After monolingual fine-tuning, does the model handle all three Konkani scripts?

### 📘️ Instruction data effect:

* How much additional gain is achieved by adding the Alpaca-style Konkani instruction set?

---

# Data Augmentation Strategy for Monolingual Fine-Tuning

For each datapoint in [`anag007/asmitai_wiki_konkani_dataset`](https://huggingface.co/datasets/anag007/asmitai_wiki_konkani_dataset), we create synthetic instruction–source–target triples using BART-style corruption functions.
While generating synthetic instruction data from this data, each article is first split into chunks capped at a maximum of 512 tokens to ensure compatibility with model context limits and efficient processing.

### Triple Format

* **Instruction**: Describes the corruption type; written in English or native script.
* **Source**: Corrupted input (e.g., masked, shuffled, deleted tokens).
* **Target**: Original clean text.

### Corruption Techniques (10–20x per example)

| Technique            | Source Corruption               | Instruction Template                                           |
| -------------------- | ------------------------------- | -------------------------------------------------------------- |
| Token masking        | Mask 10–30% tokens with `<X>`   | "Insert the correct tokens at the `<X>` positions in Konkani." |
| Token deletion       | Remove \~30% of tokens          | "Add the missing words wherever information is absent."        |
| Span infilling       | Replace token spans with `<Y>`  | "Fill each `<Y>` placeholder with the appropriate text."       |
| Sentence permutation | Shuffle sentence order          | "Re-order the sentences to make the paragraph coherent."       |
| Document rotation    | Rotate text from a random index | "Restore the original order of the passage."                   |

---

# Data Analysis

## Model Selection Based on Token Counts

We evaluated two models for their tokenization efficiency over Konkani:

* `google/gemma-3-12b-it`
* `meta-llama/Llama-3.1-8B-Instruct`

Using the [`anag007/asmitai_wiki_konkani_dataset`](https://huggingface.co/datasets/anag007/asmitai_wiki_konkani_dataset) corpus, we measured token counts via:

```bash
python data_analysis/get_total_token_count_from_data.py --model meta-llama/Llama-3.1-8B-Instruct
# Output: Number of tokens: 5648056

python data_analysis/get_total_token_count_from_data.py --model google/gemma-3-12b-it
# Output: Number of tokens: 4074954
```

### 🔹 Conclusion:

`google/gemma-3-12b-it` produces fewer tokens on Konkani data and is therefore more cost-efficient and script-aware.

---

# Released Datasets

We release two instruction-tuning datasets generated from monolingual and alpaca style translated data:


1. **Purely Noisified Instruction Data**
   `anag007/asmitai_konkani_gemma-3-12b_noisified_instruction_data`
   → Contains only synthetic instructions generated from the Asmitai monolingual dataset.
2. **Noisified Alpaca Instruction Data**
   `anag007/asmitai_konkani_gemma-3-12b_noisified_alpaca_instruction_data`
   → Contains monolingual Asmitai corpus with synthetic noise + Alpaca instruction data (MT-generated).


---

# 🪤 Noisified Data (Asmitai) Generation

## Standard Instruction Format

```python
from data_preparation.lib.prepare_noisified_data import augment_konkani_dataset
from data_preparation.lib.utils import create_dataset_dict, upload_to_huggingface

tokenizer_name = "google/gemma-3-12b-it"
repo_name = "asmitai_konkani_gemma-3-12b_noisified_instruction_data"

train_df = augment_konkani_dataset(upsample=10, tokenizer_name=tokenizer_name, type="train")
validation_df = augment_konkani_dataset(upsample=1, tokenizer_name=tokenizer_name, type="validation")
test_df = augment_konkani_dataset(upsample=1, tokenizer_name=tokenizer_name, type="test")

dataset_dict = create_dataset_dict(train_df, validation_df, test_df)
upload_to_huggingface(dataset_dict, repo_name)
```

Or run:

```bash
python data_preparation/gemma/prepare_synthetic_noisified_gemma_data.py
```

## Alpaca + Noisified Data (Asmitai) Format

```python
from data_preparation.lib.prepare_noisified_data import augment_konkani_dataset
from data_preparation.lib.utils import create_dataset_dict, upload_to_huggingface

tokenizer_name = "google/gemma-3-12b-it"
repo_name = "asmitai_konkani_gemma-3-12b_noisified_alpaca_instruction_data"

train_df = augment_konkani_dataset(upsample=10, tokenizer_name=tokenizer_name, instruction_data=True, type="train")
validation_df = augment_konkani_dataset(upsample=1, tokenizer_name=tokenizer_name, instruction_data=True, type="validation")
test_df = augment_konkani_dataset(upsample=1, tokenizer_name=tokenizer_name, instruction_data=True, type="test")

dataset_dict = create_dataset_dict(train_df, validation_df, test_df)
upload_to_huggingface(dataset_dict, repo_name)
```

Or run:

```bash
python data_preparation/gemma/prepare_synthetic_noisified_alpaca_gemma_data.py
```

---

## 📁 Download & Usage
### Noisified Data (Asmitai) Instruction Data

```python
from datasets import load_dataset

dataset = load_dataset("anag007/asmitai_konkani_gemma-3-12b_noisified_instruction_data")
```
Or visit: [https://huggingface.co/datasets/anag007/asmitai_konkani_gemma-3-12b_noisified_instruction_data](https://huggingface.co/datasets/anag007/asmitai_konkani_gemma-3-12b_noisified_instruction_data)


### Alpaca + Noisified Data (Asmitai) Instruction Data

```python
from datasets import load_dataset

dataset = load_dataset("anag007/asmitai_konkani_gemma-3-12b_noisified_alpaca_instruction_data")
```

Or visit: [https://huggingface.co/datasets/anag007/asmitai\_konkani\_gemma-3-12b\_noisified\_alpaca\_instruction\_data](https://huggingface.co/datasets/anag007/asmitai_konkani_gemma-3-12b_noisified_alpaca_instruction_data)

---

# 📊 Dataset Distribution (To Be Filled)

| Dataset Name                                                        | Total Samples | Train Samples | Validation | Test |
| ------------------------------------------------------------------- | ------------- | ------------- | ---------- | ---- |
| asmitai\_konkani\_gemma-3-12b\_noisified\_instruction\_data         |               |      96740         |     1211       |    1202  |
| asmitai\_konkani\_gemma-3-12b\_noisified\_alpaca\_instruction\_data |               |        138340       |    6411        |  6403    |

*Please update the table with actual dataset sizes after generation.*
