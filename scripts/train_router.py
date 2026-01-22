#!/usr/bin/env python3
"""Training script for the SemanticRouter model.

This script fine-tunes a DistilBERT model on the Databricks Dolly 15k dataset
for binary classification of queries into Fast Path (0) or Slow Path (1).

Features:
- Early stopping based on validation loss convergence
- Warmup + Linear decay learning rate scheduler
- Prometheus Pushgateway integration for real-time monitoring
- Gradient clipping and norm monitoring

Usage:
    python scripts/train_router.py --output-dir ./models/router

    # With Prometheus monitoring:
    python scripts/train_router.py \
        --output-dir ./models/router \
        --pushgateway-url http://localhost:9091

Category Mapping:
    - Fast Path (Label 0): classification, summarization
    - Slow Path (Label 1): creative_writing, open_qa (general_qa in dataset)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training.metrics import EarlyStoppingState, TrainingMetrics

# =============================================================================
# Constants
# =============================================================================

# Category to label mapping
CATEGORY_TO_LABEL = {
    "classification": 0,
    "summarization": 0,
    "creative_writing": 1,
    "open_qa": 1,
    "general_qa": 1,  # Alternative name in some versions
}

# Target categories from the spec
TARGET_CATEGORIES = {
    "classification",
    "summarization",
    "creative_writing",
    "open_qa",
    "general_qa",
}

# Label names for model config
LABEL_NAMES = {0: "fast_path", 1: "slow_path"}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model
    base_model: str = "distilbert-base-uncased"
    max_length: int = 512

    # Training hyperparameters
    max_epochs: int = 20  # Upper bound; early stopping will likely trigger before
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0

    # Learning rate scheduler
    warmup_ratio: float = 0.1  # 10% of total steps for warmup

    # Early stopping
    patience: int = 3
    min_delta: float = 1e-4

    # Data splits
    test_size: float = 0.1
    val_size: float = 0.1
    random_seed: int = 42

    # Output
    output_dir: str = "./models/router"
    save_total_limit: int = 3  # Keep top N checkpoints

    # Monitoring
    pushgateway_url: str | None = None
    log_interval: int = 50  # Log every N steps

    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps


@dataclass
class TrainingState:
    """Tracks training state across epochs."""

    current_epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_metrics: list[dict[str, float]] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)


# =============================================================================
# Data Loading
# =============================================================================


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
    print(
        f"Label distribution: Fast Path (0): {label_counts[0]}, "
        f"Slow Path (1): {label_counts[1]}"
    )

    return texts, labels


def create_datasets(
    texts: list[str],
    labels: list[int],
    config: TrainingConfig,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split data into train, validation, and test sets.

    Args:
        texts: List of input texts.
        labels: List of labels.
        config: Training configuration.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    # First split: separate test set
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=labels,
    )

    # Second split: separate validation set
    val_ratio = config.val_size / (1 - config.test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=val_ratio,
        random_state=config.random_seed,
        stratify=train_val_labels,
    )

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    return train_dataset, val_dataset, test_dataset


# =============================================================================
# Training Loop
# =============================================================================


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        predictions: Model predictions (logits).
        labels: Ground truth labels.

    Returns:
        Dictionary of metrics.
    """
    preds = predictions.argmax(-1).cpu().numpy()
    labels = labels.cpu().numpy()

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
    }


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: TrainingConfig,
    state: TrainingState,
    device: torch.device,
    metrics_tracker: TrainingMetrics | None = None,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        config: Training configuration.
        state: Training state.
        device: Device to train on.
        metrics_tracker: Optional metrics tracker for step-level logging.

    Returns:
        Tuple of (average_loss, average_gradient_norm).
    """
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient clipping and norm calculation
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Track metrics
        total_loss += loss.item()
        total_grad_norm += grad_norm.item()
        num_batches += 1
        state.global_step += 1

        # Step-level logging and metrics push (industry standard)
        if state.global_step % config.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  Step {state.global_step}: "
                f"loss={loss.item():.4f}, "
                f"lr={current_lr:.2e}, "
                f"grad_norm={grad_norm.item():.4f}"
            )

            # Push step-level metrics to Prometheus (real-time monitoring)
            if metrics_tracker is not None:
                metrics_tracker.log_step(
                    step=state.global_step,
                    train_loss=loss.item(),
                    learning_rate=current_lr,
                    gradient_norm=grad_norm.item(),
                )
                metrics_tracker.push()

    avg_loss = total_loss / num_batches
    avg_grad_norm = total_grad_norm / num_batches

    return avg_loss, avg_grad_norm


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate model on a dataset.

    Args:
        model: The model to evaluate.
        dataloader: Evaluation data loader.
        device: Device to evaluate on.

    Returns:
        Tuple of (average_loss, metrics_dict).
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        total_loss += outputs.loss.item()

        all_predictions.append(outputs.logits)
        all_labels.append(batch["labels"])

    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(all_predictions, all_labels)
    return avg_loss, metrics


def train(
    config: TrainingConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
) -> dict[str, Any]:
    """Main training function.

    Args:
        config: Training configuration.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.

    Returns:
        Dictionary with training results and metadata.
    """
    # Setup device
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
    print(f"Using device: {device}")

    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Initialize metrics tracking
    run_id = str(uuid.uuid4())[:8]
    metrics_tracker = TrainingMetrics(
        pushgateway_url=config.pushgateway_url,
        job_name="router_training",
        run_id=run_id,
    )
    metrics_tracker.set_training_active(True)
    metrics_tracker.push()

    # Load tokenizer and model
    print(f"Loading base model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=2,
        id2label=LABEL_NAMES,
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )
    model.to(device)

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_length,
        )

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    val_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    # Rename label to labels for HuggingFace model
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    # Create data loaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Setup learning rate scheduler with warmup
    total_steps = len(train_loader) * config.max_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # Initialize training state
    state = TrainingState()
    early_stopping = EarlyStoppingState(
        patience=config.patience,
        min_delta=config.min_delta,
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training with early stopping")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Patience: {config.patience}")
    print(f"  Min delta: {config.min_delta}")
    print("=" * 60 + "\n")

    training_start_time = time.time()

    for epoch in range(1, config.max_epochs + 1):
        state.current_epoch = epoch
        epoch_start_time = time.time()

        print(f"\nEpoch {epoch}/{config.max_epochs}")
        print("-" * 40)

        # Train (with step-level metrics pushing)
        train_loss, avg_grad_norm = train_epoch(
            model, train_loader, optimizer, scheduler, config, state, device,
            metrics_tracker=metrics_tracker,
        )

        # Evaluate
        val_loss, val_metrics = evaluate(model, val_loader, device)

        # Calculate epoch duration and throughput
        epoch_duration = time.time() - epoch_start_time
        samples_per_sec = len(train_dataset) / epoch_duration

        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]

        # Update early stopping
        is_best = early_stopping.update(val_loss)

        # Track state
        state.train_losses.append(train_loss)
        state.val_losses.append(val_loss)
        state.val_metrics.append(val_metrics)
        state.learning_rates.append(current_lr)

        if is_best:
            state.best_val_loss = val_loss
            state.best_epoch = epoch
            # Save best model
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print(f"  [NEW BEST] Saved model with val_loss={val_loss:.4f}")

        # Log metrics
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} (best: {early_stopping.best_loss:.4f})")
        print(
            f"  Val Metrics: acc={val_metrics['accuracy']:.4f}, "
            f"f1={val_metrics['f1']:.4f}, "
            f"prec={val_metrics['precision']:.4f}, "
            f"rec={val_metrics['recall']:.4f}"
        )
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  Gradient Norm: {avg_grad_norm:.4f}")
        print(f"  Throughput: {samples_per_sec:.1f} samples/sec")
        print(
            f"  Early Stopping: patience {early_stopping.counter}/{config.patience}"
        )

        # Push metrics to Prometheus
        metrics_tracker.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_metrics["accuracy"],
            val_f1=val_metrics["f1"],
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
            learning_rate=current_lr,
            best_val_loss=early_stopping.best_loss,
            patience_counter=early_stopping.counter,
            is_best=is_best,
            global_step=state.global_step,
            gradient_norm=avg_grad_norm,
            throughput=samples_per_sec,
        )
        metrics_tracker.push()

        # Check early stopping
        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            metrics_tracker.set_early_stopped(True)
            break

    training_duration = time.time() - training_start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {training_duration:.1f}s")
    print(f"Best model from epoch {state.best_epoch} with val_loss={state.best_val_loss:.4f}")
    print("=" * 60)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    # Reload best model for test evaluation
    model = AutoModelForSequenceClassification.from_pretrained(str(output_dir))
    model.to(device)
    test_loss, test_metrics = evaluate(model, test_loader, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(
        f"  Accuracy: {test_metrics['accuracy']:.4f}, "
        f"F1: {test_metrics['f1']:.4f}, "
        f"Precision: {test_metrics['precision']:.4f}, "
        f"Recall: {test_metrics['recall']:.4f}"
    )

    # Mark training as complete
    metrics_tracker.set_training_active(False)
    metrics_tracker.push()

    # Save training metadata
    metadata = {
        "run_id": run_id,
        "base_model": config.base_model,
        "max_epochs": config.max_epochs,
        "actual_epochs": state.current_epoch,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "warmup_ratio": config.warmup_ratio,
        "patience": config.patience,
        "min_delta": config.min_delta,
        "gradient_clip_norm": config.gradient_clip_norm,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_epoch": state.best_epoch,
        "best_val_loss": state.best_val_loss,
        "early_stopped": early_stopping.should_stop,
        "training_duration_seconds": training_duration,
        "device": str(device),
        "test_metrics": test_metrics,
        "training_history": {
            "train_losses": state.train_losses,
            "val_losses": state.val_losses,
            "val_metrics": state.val_metrics,
            "learning_rates": state.learning_rates,
        },
        "category_mapping": CATEGORY_TO_LABEL,
    }

    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel and metadata saved to: {output_dir}")

    return metadata


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train SemanticRouter model with early stopping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/router",
        help="Output directory for the trained model",
    )

    # Model
    parser.add_argument(
        "--base-model",
        type=str,
        default="distilbert-base-uncased",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    # Training hyperparameters
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs (early stopping may trigger before)",
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
        help="Initial learning rate",
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Epochs to wait for improvement before stopping",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum change to qualify as improvement",
    )

    # Learning rate scheduler
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps for warmup",
    )

    # Monitoring
    parser.add_argument(
        "--pushgateway-url",
        type=str,
        default=None,
        help="Prometheus Pushgateway URL (e.g., http://localhost:9091)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log training metrics every N steps",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for training",
    )

    args = parser.parse_args()

    # Create config from arguments
    config = TrainingConfig(
        output_dir=args.output_dir,
        base_model=args.base_model,
        max_length=args.max_length,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        min_delta=args.min_delta,
        warmup_ratio=args.warmup_ratio,
        pushgateway_url=args.pushgateway_url,
        log_interval=args.log_interval,
        device=args.device,
    )

    # Load and prepare data
    texts, labels = load_and_filter_data()
    train_dataset, val_dataset, test_dataset = create_datasets(texts, labels, config)

    # Train
    metadata = train(config, train_dataset, val_dataset, test_dataset)

    print("\nTraining complete!")
    print(f"  Best epoch: {metadata['best_epoch']}")
    print(f"  Best val loss: {metadata['best_val_loss']:.4f}")
    print(f"  Test F1: {metadata['test_metrics']['f1']:.4f}")
    print(f"  Early stopped: {metadata['early_stopped']}")


if __name__ == "__main__":
    main()
