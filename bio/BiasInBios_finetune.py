import pickle
import json
import argparse
import numpy as np
from datasets import Dataset, load_metric
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# Load the dataset from the pickle file
def load_data(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        data = pickle.load(f)
    return data


# Load the mappings from JSON
def load_mappings(json_file_path):
    with open(json_file_path, "r") as f:
        mappings = json.load(f)
    return mappings


# Preprocess the dataset using mappings
def preprocess_data(data, title_mapping):
    inputs = [item["raw"] for item in data]
    labels = [
        title_mapping[item["title"]] for item in data if item["title"] in title_mapping
    ]
    return inputs, labels


# Tokenize inputs
def tokenize_data(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Dataset class (using Huggingface datasets)
def create_hf_dataset(inputs, labels):
    # Create a Huggingface Dataset from the input and label lists
    dataset = Dataset.from_dict({"text": inputs, "label": labels})
    return dataset


metric = load_metric("accuracy", trust_remote_code=True)


def comp_accuracy(pred):
    # Extract logits and labels
    logits, labels = pred
    # Convert logits to predictions (argmax on logits for classification)
    predictions = np.argmax(logits, axis=-1)
    # Compute accuracy
    accuracy = metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy}


# Fine-tuning the model
def fine_tune_model(train_dataset, valid_dataset, model_name, batch_size, num_epochs):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(set(labels)), trust_remote_code=True
    )

    training_args = TrainingArguments(
        output_dir=f"data/Finetuned_Models/finetuned-{model_name.replace('/', '--')}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        auto_find_batch_size=True,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=1000,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=comp_accuracy,
    )

    trainer.train()


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Fine-tune a Huggingface model with BiasBios dataset"
    )

    # Adding arguments for the input files and model
    parser.add_argument(
        "--data_path",
        type=str,
        default="BIOS.pkl",
        help="Path to the pickle file containing the dataset",
    )
    parser.add_argument(
        "--mappings_file",
        type=str,
        default="data/MAPPINGS.json",
        help="Path to the JSON file containing the mappings",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Huggingface model to fine-tune"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of epochs for training"
    )

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    wandb.login()

    # Load and preprocess data
    data = load_data(args.data_path)
    mappings = load_mappings(args.mappings_file)

    # Extract title mapping from the loaded mappings
    title_mapping = mappings["title_mapping"]

    # Preprocess the data
    inputs, labels = preprocess_data(data, title_mapping)

    # Tokenizer based on the selected model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create a Huggingface Dataset
    dataset = create_hf_dataset(inputs, labels)

    # Tokenize the Huggingface Dataset
    tokenized_dataset = dataset.map(
        tokenize_data,
        batched=True,
        cache_file_name=f"data/Finetuned_Models/finetuned-{args.model_name.replace('/', '--')}/tokenized_dataset",
        keep_in_memory=False,
    )
    splitted_ds = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

    # Fine-tune the model
    fine_tune_model(
        splitted_ds["train"],
        splitted_ds["test"],
        args.model_name,
        args.batch_size,
        args.num_epochs,
    )
