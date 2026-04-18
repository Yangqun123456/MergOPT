import os
import json
from datasets import Dataset


def load_local_dataset(dataset_name, split='train'):
    try:
        dataset_path = os.path.join('datasets', dataset_name, f'{split}.json')

        if not os.path.exists(dataset_path):
            print(f"Dataset file not found: {dataset_path}")
            return None

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return Dataset.from_list(data)
        elif isinstance(data, dict) and "examples" in data:
            return Dataset.from_list(data["examples"])
        elif isinstance(data, dict):
            return Dataset.from_dict(data)

        print(f"Unsupported dataset format: {dataset_path}")
        return None

    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        return None


def prepare_dataset_for_training(dataset, dataset_name=None):
    if dataset is None:
        return None

    columns = dataset.column_names
    if "prompt" in columns and "answer" in columns:
        def convert_to_text_format(example):
            return {"text": f"{example['prompt']}\n{example['answer']}"}

        processed_dataset = dataset.map(convert_to_text_format)
        return processed_dataset
    else:
        print(f"Warning: Dataset {dataset_name} does not have standard prompt-answer format, trying other conversion methods")

        def fallback_conversion(example):
            text_parts = []
            for key, value in example.items():
                if isinstance(value, str) and value.strip():
                    text_parts.append(f"{key}: {value}")

            return {"text": "\n".join(text_parts)}

        processed_dataset = dataset.map(fallback_conversion)
        return processed_dataset


def get_available_datasets():
    datasets_dir = 'datasets'
    if not os.path.exists(datasets_dir):
        print(f"Datasets directory not found")
        return []

    return [name for name in os.listdir(datasets_dir)
            if os.path.isdir(os.path.join(datasets_dir, name))]