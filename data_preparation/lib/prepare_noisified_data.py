import random
import re
from random import shuffle
from re import split

from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


# Constants
MAX_TOKENS = 512
CORRUPTION_FUNCTIONS = ["mask", "delete", "infill", "permute", "rotate"]

# Instruction templates per corruption type and script
instruction_info = {
    "mask": {
        "eng": "Insert the correct tokens at the <X> positions in Konkani.",
        "dev": "कोंकणींतल्या <X> स्थानांचेर योग्य टोकन घालचे.",
        "rom": "Konknnintlea <X> sthanancher yogy ttokon ghalche.",
        "kan": "ಕೊಂಕ್ಣೀಂತ್ಲ್ಯಾ <X> ಸ್ಥಾನಾಂಚೆರ್ ಯೊಗ್ಯ್ ಟೊಕನ್ ಘಾಲ್ಚೆ.",
        "mal": "കൊങ്ക്ണീന്ത്ല്യാ <X> സ്ഥാനാഞ്ചെർ യൊഗ്യ് ടൊകൻ ഘാൽചെ."
    },
    "delete": {
        "eng": "Add the missing words wherever information is absent in Konkani.",
        "dev": "कोंकणींत जंय जंय म्हायती ना थंय थंय गळून पडपी उतरां जोडात.",
        "rom": "Konknnint zom-i zom-i mhaiti na thoim thoim gollun poddpi utram zoddat.",
        "kan": "ಕೊಂಕ್ಣೀಂತ್ ಜಂಯ್ ಜಂಯ್ ಮ್ಹಾಯ್ತಿ ನಾ ಥಂಯ್ ಥಂಯ್ ಗಳೂನ್ ಪಡ್ಪಿ ಉತ್ರಾಂ ಜೊಡಾತ್.",
        "mal": "കൊങ്ക്ണീന്ത് ജംയ് ജംയ് മ്ഹായ്തി നാ ഥംയ് ഥംയ് ഗളൂൻ പഡ്പി ഉത്രാം ജൊഡാത്."
    },
    "infill": {
      "eng": "Fill each <Y> placeholder with the appropriate text in Konkani.",
        "dev": "दरेक <Y> प्लेसहोल्डर कोंकणींतल्यान फावो तो मजकूर भरचो.",
        "rom": "Dorek <Y> plesholddor konknnintlean favo to mojkur bhorcho.",
        "kan": "ದರೆಕ್ <Y> ಪ್ಲೆಸ್ಹೊಲ್ಡರ್ ಕೊಂಕ್ಣೀಂತ್ಲ್ಯಾನ್ ಫಾವೊ ತೊ ಮಜ್ಕೂರ್ ಭರ್ಚೊ.",
        "mal": "ദരെക് <Y> പ്ലെസ്ഹൊൽഡർ കൊങ്ക്ണീന്ത്ല്യാൻ ഫാവൊ തൊ മജ്കൂർ ഭർചൊ."
    },
    "permute": {
      "eng": "Re-order the sentences so the paragraph reads naturally.",
        "dev": "परिच्छेद सैमीक रितीन वाचपाक मेळचो म्हणून वाक्यांची परतून क्रमवारी करची.",
        "rom": "Porichchhed soimik ritin vachpak mellcho mhonnun vakeanchi portun kromvari korchi.",
        "kan": "ಪರಿಚ್ಛೆದ್ ಸೈಮೀಕ್ ರಿತೀನ್ ವಾಚ್ಪಾಕ್ ಮೆಳ್ಚೊ ಮ್ಹಣೂನ್ ವಾಕ್ಯಾಂಚಿ ಪರ್ತೂನ್ ಕ್ರಮ್ವಾರಿ ಕರ್ಚಿ.",
        "mal": "പരിച്ഛെദ് സൈമീക് രിതീൻ വാച്പാക് മെൾചൊ മ്ഹണൂൻ വാക്യാഞ്ചി പർതൂൻ ക്രമ്വാരി കർചി."
    },
    "rotate": {
      "eng": "Restore the original sentence order of the passage.",
        "dev": "उताऱ्याचो मूळ वाक्य क्रम परतून हाडचो.",
        "rom": "Utareacho mull vaky krom' portun haddcho.",
        "kan": "ಉತಾರ್ಯಾಚೊ ಮೂಳ್ ವಾಕ್ಯ್ ಕ್ರಮ್ ಪರ್ತೂನ್ ಹಾಡ್ಚೊ.",
        "mal": "ഉതാര്യാചൊ മൂൾ വാക്യ് ക്രം പർതൂൻ ഹാഡ്ചൊ."
    }
}

random.seed(42)

def detect_scripts(text):
    """
    Detects script types used in a given text.
    :param text:
    :return:
    """
    script_patterns = {
        'dev': r'[\u0900-\u097F]',
        'rom': r'[A-Za-z]',
        'kan': r'[\u0C80-\u0CFF]',
        'mal': r'[\u0D00-\u0D7F]',
    }
    return [s for s, pat in script_patterns.items() if re.search(pat, text)]

def chunk_by_sentence(text, tokenizer, max_tokens=512):
    """
    Splits text into token-limited chunks based on sentence boundaries.
    :param text:
    :param tokenizer:
    :param max_tokens:
    :return:
    """
    sentences = re.split(r'(?<=[।.!?])\s+', text.strip())
    chunks, current_chunk, current_token_count = [], "", 0

    for sentence in sentences:
        sentence = sentence.strip()
        token_count = len(tokenizer.tokenize(sentence))
        if token_count > max_tokens:
            continue
        if current_token_count + token_count <= max_tokens:
            current_chunk += sentence + " "
            current_token_count += token_count
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk, current_token_count = sentence + " ", token_count

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Corruption Functions
def mask_tokens(text, tokenizer, mask_token="<X>", min_ratio=0.1, max_ratio=0.3):
    tokens = tokenizer.tokenize(text)
    ratio = random.uniform(min_ratio, max_ratio)
    indices = random.sample(range(len(tokens)), max(1, int(len(tokens) * ratio)))
    for i in indices:
        tokens[i] = mask_token
    return tokenizer.convert_tokens_to_string(tokens)

def delete_tokens(text, tokenizer, ratio=0.3):
    tokens = tokenizer.tokenize(text)
    indices = sorted(random.sample(range(len(tokens)), max(1, int(len(tokens) * ratio))), reverse=True)
    for i in indices:
        del tokens[i]
    return tokenizer.convert_tokens_to_string(tokens)

def infill_spans(text, tokenizer, infill_token="<Y>", ratio=0.10):
    tokens = tokenizer.tokenize(text)
    total_tokens = len(tokens)

    if total_tokens < 10:
        return text

    # Compute how many tokens we want to replace (10%)
    tokens_to_replace = max(1, int(ratio * total_tokens))
    replaced = 0
    used_indices = set()

    while replaced < tokens_to_replace and len(tokens) >= 10:
        # Pick a valid span length
        span_len = min(random.randint(5, 10), tokens_to_replace - replaced)

        # Make sure span fits
        max_start = len(tokens) - span_len
        if max_start <= 0:
            break

        start = random.randint(0, max_start)
        end = start + span_len

        # Avoid overlapping spans
        if any(i in used_indices for i in range(start, end)):
            continue

        tokens[start:end] = [infill_token]
        used_indices.update(range(start, end))
        replaced += span_len

    return tokenizer.convert_tokens_to_string(tokens)

def permute_sentences(text, tokenizer):
    sentences = re.split(r'(?<=[।.!?])\s+', text.strip())
    if len(sentences) < 2:
        return text
    random.shuffle(sentences)
    return " ".join(sentences)

def rotate_document(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    k = random.randint(1, len(tokens) - 1)
    return tokenizer.convert_tokens_to_string(tokens[k:] + tokens[:k])

# Mapping function names
corruption_fn_map = {
    "mask": mask_tokens,
    "delete": delete_tokens,
    "infill": infill_spans,
    "permute": permute_sentences,
    "rotate": rotate_document
}

def create_random_augmented_triples(text, tokenizer):
    """
    Creates a list of augmented triples using a randomly chosen corruption technique.
    :param text:
    :param tokenizer:
    :return:
    """
    triples = []
    method = random.choice(CORRUPTION_FUNCTIONS)
    scripts = detect_scripts(text)
    lang = random.choice(["eng", scripts[0]]) if scripts else "eng"
    instruction = instruction_info[method][lang]
    corrupted = corruption_fn_map[method](text, tokenizer)
    triples.append({"instruction": instruction, "input": corrupted, "output": text})
    return triples


def augment_konkani_dataset(upsample=3,  tokenizer_name="google/gemma-3-1b-it", instruction_data=False , type="train"):
    """
        Main function to augment Konkani dataset.


    :param upsample: Number of time augmentation needs to be upsampled
    :param tokenizer_name: Name of the model that needs to be tokenized
    :param type: train, validation and test
    :return:        pd.DataFrame: Augmented dataset in DataFrame format.

    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = load_dataset("anag007/asmitai_wiki_konkani_dataset", split=type)
    instruct_parts = []
    if instruction_data:

        if type == "train":
            instruction_dataset = load_dataset("saillab/alpaca-konkani-cleaned", split=type)

            data_list = list(instruction_dataset)
            chunk_size = len(data_list) // upsample

            instruct_parts = [Dataset.from_list(data_list[i * chunk_size: (i + 1) * chunk_size]) for i in range(upsample)]
        elif type == "validation":
            instruction_dataset = load_dataset("saillab/alpaca-konkani-cleaned", split="test")
            total_len = len(instruction_dataset)
            mid = total_len // 2
            instruct_parts = [instruction_dataset.select(range(0, mid))] * upsample
        elif type == "test":
            instruction_dataset = load_dataset("saillab/alpaca-konkani-cleaned", split="test")
            total_len = len(instruction_dataset)
            mid = total_len // 2
            instruct_parts = [instruction_dataset.select(range(mid, total_len))] * upsample

    augmented_data = []

    for _ in range(upsample):
        data = []
        for i, sample in enumerate(tqdm(dataset, desc="Processing dataset")):
            if not sample["text"].strip():
                continue
            for chunk in chunk_by_sentence(sample["text"], tokenizer):
                data.extend(create_random_augmented_triples(chunk, tokenizer))
        if instruction_data:
            data.extend(instruct_parts[_])
        shuffle(data)
        augmented_data.extend(data)



    df = pd.DataFrame(augmented_data)
    # df.to_csv(save_path, index=False)
    return df



