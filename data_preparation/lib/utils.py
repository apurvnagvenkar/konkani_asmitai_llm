from datasets import Dataset, DatasetDict

def create_dataset_dict(train_data, val_data, test_data):
    """
    Creates a Hugging Face DatasetDict from train, validation, and test datasets.

    Args:
        train_data (list): Training data.
        val_data (list): Validation data.
        test_data (list): Test data.

    Returns:
        DatasetDict: Hugging Face dataset dictionary.
    """
    print(f"Creating DatasetDict...")
    print(val_data)
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_data),
        'validation': Dataset.from_pandas(val_data),
        'test': Dataset.from_pandas(test_data)
    })
    print(f"DatasetDict created with splits: {dataset_dict}")
    return dataset_dict


def upload_to_huggingface(dataset_dict, repo_name):
    """
    Uploads the dataset to the Hugging Face Hub.

    Args:
        dataset_dict (DatasetDict): The dataset to upload.
        repo_name (str): The Hugging Face repository name.

    Returns:
        None
    """
    print(f"Uploading dataset to Hugging Face Hub under {repo_name}...")
    dataset_dict.push_to_hub(repo_name)
    print(f"Dataset uploaded successfully.")

