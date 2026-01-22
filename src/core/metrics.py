"""Prometheus metrics definitions for observability.

Metrics follow the naming convention: {namespace}_{subsystem}_{name}_{unit}
Reference: https://prometheus.io/docs/practices/naming/
"""

from prometheus_client import Counter, Gauge, Histogram

# Namespace for all metrics
NAMESPACE = "query_gateway"

# Request metrics
REQUEST_COUNT = Counter(
    name="requests_total",
    documentation="Total number of classification requests",
    labelnames=["status", "cache_hit"],
    namespace=NAMESPACE,
)

REQUEST_LATENCY = Histogram(
    name="request_latency_seconds",
    documentation="Request latency in seconds",
    labelnames=["endpoint"],
    namespace=NAMESPACE,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Inference metrics
INFERENCE_LATENCY = Histogram(
    name="inference_latency_seconds",
    documentation="Model inference latency in seconds",
    namespace=NAMESPACE,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

INFERENCE_BATCH_SIZE = Histogram(
    name="inference_batch_size",
    documentation="Batch size for inference operations",
    namespace=NAMESPACE,
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)

# Classification metrics
CLASSIFICATION_COUNT = Counter(
    name="classifications_total",
    documentation="Total classifications by label",
    labelnames=["label"],
    namespace=NAMESPACE,
)

CONFIDENCE_SCORE = Histogram(
    name="confidence_score",
    documentation="Distribution of confidence scores",
    labelnames=["label"],
    namespace=NAMESPACE,
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
)

CONFIDENCE_ROUTING_COUNT = Counter(
    name="confidence_routing_total",
    documentation="Requests routed to slow path due to low confidence",
    labelnames=["original_label", "routed_label"],
    namespace=NAMESPACE,
)

# Cache metrics
CACHE_HIT_COUNT = Counter(
    name="cache_hits_total",
    documentation="Total cache hits",
    labelnames=["level"],
    namespace=NAMESPACE,
)

CACHE_MISS_COUNT = Counter(
    name="cache_misses_total",
    documentation="Total cache misses",
    labelnames=["level"],
    namespace=NAMESPACE,
)

CACHE_SIZE = Gauge(
    name="cache_size",
    documentation="Current cache size (number of entries)",
    labelnames=["level"],
    namespace=NAMESPACE,
)

# Batching metrics
BATCH_QUEUE_SIZE = Gauge(
    name="batch_queue_size",
    documentation="Current number of requests waiting in batch queue",
    namespace=NAMESPACE,
)

BATCH_WAIT_TIME = Histogram(
    name="batch_wait_time_seconds",
    documentation="Time spent waiting in batch queue",
    namespace=NAMESPACE,
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05),
)

# System health metrics
MODEL_LOADED = Gauge(
    name="model_loaded",
    documentation="Whether the model is loaded and ready (1=ready, 0=not ready)",
    namespace=NAMESPACE,
)

ACTIVE_REQUESTS = Gauge(
    name="active_requests",
    documentation="Number of requests currently being processed",
    namespace=NAMESPACE,
)
