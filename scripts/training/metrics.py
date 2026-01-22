"""Prometheus metrics for model training.

This module provides metrics integration with Prometheus Pushgateway
for monitoring model training progress in real-time.

Industry-standard approach:
- Step-level metrics: train_loss, learning_rate, gradient_norm (pushed every N steps)
- Epoch-level metrics: val_loss, val_f1, etc. (pushed after each epoch)

Usage:
    metrics = TrainingMetrics(pushgateway_url="http://localhost:9091")

    # During training loop (every N steps):
    metrics.log_step(step=100, train_loss=0.5, learning_rate=2e-5, gradient_norm=0.8)
    metrics.push()

    # After each epoch:
    metrics.log_epoch(epoch=1, val_loss=0.4, ...)
    metrics.push()
"""

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


# Training job namespace
NAMESPACE = "router_training"


@dataclass
class TrainingMetrics:
    """Prometheus metrics collector for model training.

    Attributes:
        pushgateway_url: URL of the Prometheus Pushgateway.
        job_name: Name for this training job (used as job label).
        run_id: Unique identifier for this training run.
    """

    pushgateway_url: str | None = None
    job_name: str = "router_training"
    run_id: str = "default"

    # Internal state
    _registry: CollectorRegistry = field(default_factory=CollectorRegistry, repr=False)
    _gauges: dict[str, Gauge] = field(default_factory=dict, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize Prometheus gauges."""
        self._init_gauges()

    def _init_gauges(self) -> None:
        """Create all training metric gauges."""
        if self._initialized:
            return

        # =================================================================
        # Step-level metrics (updated every N steps during training)
        # =================================================================
        self._gauges["global_step"] = Gauge(
            "global_step",
            "Global training step",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["step_train_loss"] = Gauge(
            "step_train_loss",
            "Training loss at current step (real-time)",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["step_learning_rate"] = Gauge(
            "step_learning_rate",
            "Learning rate at current step",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["step_gradient_norm"] = Gauge(
            "step_gradient_norm",
            "Gradient norm at current step",
            namespace=NAMESPACE,
            registry=self._registry,
        )

        # =================================================================
        # Epoch-level metrics (updated after each epoch)
        # =================================================================
        self._gauges["epoch"] = Gauge(
            "epoch",
            "Current training epoch",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["epoch_train_loss"] = Gauge(
            "epoch_train_loss",
            "Average training loss for the epoch",
            namespace=NAMESPACE,
            registry=self._registry,
        )

        # Keep old name for backwards compatibility
        self._gauges["train_loss"] = Gauge(
            "train_loss",
            "Training loss (epoch average)",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["val_loss"] = Gauge(
            "val_loss",
            "Validation loss",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["best_val_loss"] = Gauge(
            "best_val_loss",
            "Best validation loss achieved",
            namespace=NAMESPACE,
            registry=self._registry,
        )

        # Performance metrics
        self._gauges["val_accuracy"] = Gauge(
            "val_accuracy",
            "Validation accuracy",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["val_f1"] = Gauge(
            "val_f1",
            "Validation F1 score",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["val_precision"] = Gauge(
            "val_precision",
            "Validation precision",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["val_recall"] = Gauge(
            "val_recall",
            "Validation recall",
            namespace=NAMESPACE,
            registry=self._registry,
        )

        # Learning rate
        self._gauges["learning_rate"] = Gauge(
            "learning_rate",
            "Current learning rate",
            namespace=NAMESPACE,
            registry=self._registry,
        )

        # Training health
        self._gauges["gradient_norm"] = Gauge(
            "gradient_norm",
            "Gradient norm (for detecting explosion/vanishing)",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["throughput_samples_per_sec"] = Gauge(
            "throughput_samples_per_sec",
            "Training throughput in samples per second",
            namespace=NAMESPACE,
            registry=self._registry,
        )

        # Early stopping state
        self._gauges["patience_counter"] = Gauge(
            "patience_counter",
            "Epochs without improvement",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["is_best_model"] = Gauge(
            "is_best_model",
            "Whether current epoch produced best model (1=yes, 0=no)",
            namespace=NAMESPACE,
            registry=self._registry,
        )

        # Training status
        self._gauges["training_active"] = Gauge(
            "training_active",
            "Whether training is currently running (1=yes, 0=no)",
            namespace=NAMESPACE,
            registry=self._registry,
        )
        self._gauges["early_stopped"] = Gauge(
            "early_stopped",
            "Whether training was early stopped (1=yes, 0=no)",
            namespace=NAMESPACE,
            registry=self._registry,
        )

        self._initialized = True

    def set(self, name: str, value: float) -> None:
        """Set a metric value.

        Args:
            name: Metric name (without namespace).
            value: Metric value.
        """
        if name in self._gauges:
            self._gauges[name].set(value)

    def log_step(
        self,
        step: int,
        train_loss: float,
        learning_rate: float,
        gradient_norm: float | None = None,
    ) -> None:
        """Log step-level metrics (called every N steps during training).

        This provides real-time visibility into training progress without
        waiting for epoch completion.

        Args:
            step: Global training step.
            train_loss: Current batch/step training loss.
            learning_rate: Current learning rate.
            gradient_norm: Gradient norm for this step (optional).
        """
        self.set("global_step", step)
        self.set("step_train_loss", train_loss)
        self.set("step_learning_rate", learning_rate)
        if gradient_norm is not None:
            self.set("step_gradient_norm", gradient_norm)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        val_f1: float,
        val_precision: float,
        val_recall: float,
        learning_rate: float,
        best_val_loss: float,
        patience_counter: int,
        is_best: bool,
        global_step: int | None = None,
        gradient_norm: float | None = None,
        throughput: float | None = None,
    ) -> None:
        """Log metrics for a training epoch.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.
            val_accuracy: Validation accuracy.
            val_f1: Validation F1 score.
            val_precision: Validation precision.
            val_recall: Validation recall.
            learning_rate: Current learning rate.
            best_val_loss: Best validation loss so far.
            patience_counter: Epochs without improvement.
            is_best: Whether this epoch produced the best model.
            global_step: Global training step (optional).
            gradient_norm: Gradient norm (optional).
            throughput: Training throughput in samples/sec (optional).
        """
        self.set("epoch", epoch)
        self.set("train_loss", train_loss)
        self.set("epoch_train_loss", train_loss)  # Also set epoch-specific metric
        self.set("val_loss", val_loss)
        self.set("val_accuracy", val_accuracy)
        self.set("val_f1", val_f1)
        self.set("val_precision", val_precision)
        self.set("val_recall", val_recall)
        self.set("learning_rate", learning_rate)
        self.set("best_val_loss", best_val_loss)
        self.set("patience_counter", patience_counter)
        self.set("is_best_model", 1.0 if is_best else 0.0)

        if global_step is not None:
            self.set("global_step", global_step)
        if gradient_norm is not None:
            self.set("gradient_norm", gradient_norm)
        if throughput is not None:
            self.set("throughput_samples_per_sec", throughput)

    def set_training_active(self, active: bool) -> None:
        """Set training active status."""
        self.set("training_active", 1.0 if active else 0.0)

    def set_early_stopped(self, stopped: bool) -> None:
        """Set early stopped status."""
        self.set("early_stopped", 1.0 if stopped else 0.0)

    def push(self) -> bool:
        """Push metrics to Pushgateway.

        Returns:
            True if push was successful, False otherwise.
        """
        if not self.pushgateway_url:
            return False

        try:
            push_to_gateway(
                gateway=self.pushgateway_url,
                job=self.job_name,
                registry=self._registry,
                grouping_key={"run_id": self.run_id},
            )
            return True
        except Exception as e:
            print(f"Failed to push metrics to Pushgateway: {e}")
            return False

    def get_metrics_dict(self) -> dict[str, Any]:
        """Get all current metric values as a dictionary.

        Returns:
            Dictionary of metric names to values.
        """
        result = {}
        for name, gauge in self._gauges.items():
            # Get the current value from the gauge
            # Note: This is a simplified approach; actual value extraction
            # depends on prometheus_client internals
            try:
                result[name] = gauge._value.get()
            except Exception:
                result[name] = None
        return result


@dataclass
class EarlyStoppingState:
    """Tracks early stopping state.

    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        best_loss: Best validation loss observed.
        counter: Epochs without improvement.
        should_stop: Whether training should stop.
    """

    patience: int = 3
    min_delta: float = 1e-4
    best_loss: float = field(default=float("inf"))
    counter: int = 0
    should_stop: bool = False

    def update(self, val_loss: float) -> bool:
        """Update early stopping state with new validation loss.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if this is the best model so far, False otherwise.
        """
        is_best = False

        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            is_best = True
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return is_best

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False
