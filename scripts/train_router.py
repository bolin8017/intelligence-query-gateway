#!/usr/bin/env python3
"""Training script for the SemanticRouter model.

This script fine-tunes a DistilBERT model on the Databricks Dolly 15k dataset
for binary classification of queries into Fast Path (0) or Slow Path (1).

Usage:
    python scripts/train_router.py --output-dir ./models/router

Category Mapping:
    - Fast Path (Label 0): classification, summarization
    - Slow Path (Label 1): creative_writing, open_qa (general_qa in dataset)
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Category to label mapping
CATEGORY_TO_LABEL = {
    "classification": 0,
    "summarization": 0,
    "creative_writing": 1,
    "open_qa": 1,
    "general_qa": 1,  # Alternative name in some versions
}

# Target categories from the spec
TARGET_CATEGORIES = {"classification", "summarization", "creative_writing", "open_qa", "general_qa"}


def load_and_filter_data() -> tuple[list[str], list[int]]:
    """Load Dolly 15k dataset and filter to target categories.

    Returns:
        Tuple of (texts, labels).
    """
    print("Loading databricks/databricks-dolly-15k dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    texts = []
    labels = []

    for item in dataset:
        category = item["category"]
        if category in TARGET_CATEGORIES:
            # Use instruction as the query text
            # Combine instruction and context if context exists
            text = item["instruction"]
            if item.get("context"):
                text = f"{text}\n\nContext: {item['context']}"

            texts.append(text)
            labels.append(CATEGORY_TO_LABEL[category])

    print(f"Filtered {len(texts)} samples from target categories")

    # Print category distribution
    label_counts = {0: 0, 1: 0}
    for label in labels:
        label_counts[label] += 1
    print(f"Label distribution: Fast Path (0): {label_counts[0]}, Slow Path (1): {label_counts[1]}")

    return texts, labels


def create_datasets(
    texts: list[str],
    labels: list[int],
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split data into train, validation, and test sets.

    Args:
        texts: List of input texts.
        labels: List of labels.
        test_size: Fraction for test set.
        val_size: Fraction for validation set (from remaining after test).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    # First split: separate test set
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Second split: separate validation set
    val_ratio = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val_labels,
    )

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred) -> dict:
    """Compute evaluation metrics for the trainer.

    Args:
        eval_pred: Tuple of (predictions, labels).

    Returns:
        Dictionary of metrics.
    """
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train SemanticRouter model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/router",
        help="Output directory for the trained model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="distilbert-base-uncased",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    texts, labels = load_and_filter_data()

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(texts, labels)

    # Load tokenizer and model
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "fast_path", 1: "slow_path"},
        label2id={"fast_path": 0, "slow_path": 1},
    )

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
        )

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,  # More frequent logging for better visibility
        logging_first_step=True,
        report_to=[],  # Disable wandb/tensorboard
        dataloader_num_workers=2,  # Parallel data loading
        dataloader_pin_memory=True,  # Faster GPU transfer
        fp16=torch.cuda.is_available(),  # Mixed precision for faster training
        seed=42,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # Use new parameter name to avoid warning
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")

    # Save the best model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training metadata
    metadata = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "test_metrics": {k.replace("eval_", ""): v for k, v in test_results.items()},
        "category_mapping": CATEGORY_TO_LABEL,
    }

    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
