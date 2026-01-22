"""Prometheus metrics collection utilities for load testing.

This module provides functions to collect and analyze Prometheus metrics
from the Query Gateway service during load testing.
"""

import re
from typing import Dict, Optional

import requests


def collect_prometheus_metrics(
    url: str = "http://localhost:8000/metrics",
    timeout: int = 5,
) -> Dict[str, float]:
    """Collect metrics from Prometheus endpoint.

    Args:
        url: Prometheus metrics endpoint URL.
        timeout: Request timeout in seconds.

    Returns:
        Dictionary mapping metric names to their values.

    Raises:
        requests.RequestException: If the request fails.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to collect metrics from {url}: {e}")
        return {}

    metrics = {}

    for line in response.text.split("\n"):
        # Skip comments and empty lines
        if line.startswith("#") or not line.strip():
            continue

        # Parse Prometheus format: metric_name{labels} value
        # Simplified regex that handles metrics with or without labels
        match = re.match(r'([a-z_][a-z0-9_]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)', line)
        if match:
            metric_name, value = match.groups()
            try:
                metrics[metric_name] = float(value)
            except ValueError:
                continue

    return metrics


def calculate_cache_hit_rate(metrics: Dict[str, float]) -> float:
    """Calculate cache hit rate from metrics.

    Args:
        metrics: Dictionary of Prometheus metrics.

    Returns:
        Cache hit rate as a percentage (0-100).
    """
    hits = metrics.get("query_gateway_cache_hits_total", 0)
    misses = metrics.get("query_gateway_cache_misses_total", 0)
    total = hits + misses

    if total == 0:
        return 0.0

    return (hits / total) * 100


def calculate_average_batch_size(metrics: Dict[str, float]) -> float:
    """Calculate average batch size from metrics.

    Args:
        metrics: Dictionary of Prometheus metrics.

    Returns:
        Average batch size, or 0 if no batches processed.
    """
    batch_count = metrics.get("query_gateway_batch_count_total", 0)
    batch_sum = metrics.get("query_gateway_batch_size_sum", 0)

    if batch_count == 0:
        return 0.0

    return batch_sum / batch_count


def get_metrics_summary(
    url: str = "http://localhost:8000/metrics",
) -> Optional[Dict[str, any]]:
    """Get a summary of key performance metrics.

    Args:
        url: Prometheus metrics endpoint URL.

    Returns:
        Dictionary containing summarized metrics, or None if collection fails.
    """
    metrics = collect_prometheus_metrics(url)

    if not metrics:
        return None

    return {
        "total_requests": metrics.get("query_gateway_requests_total", 0),
        "cache_hits": metrics.get("query_gateway_cache_hits_total", 0),
        "cache_misses": metrics.get("query_gateway_cache_misses_total", 0),
        "cache_hit_rate": calculate_cache_hit_rate(metrics),
        "average_batch_size": calculate_average_batch_size(metrics),
        "p50_latency_ms": metrics.get("query_gateway_request_duration_seconds_p50", 0) * 1000,
        "p95_latency_ms": metrics.get("query_gateway_request_duration_seconds_p95", 0) * 1000,
        "p99_latency_ms": metrics.get("query_gateway_request_duration_seconds_p99", 0) * 1000,
    }


if __name__ == "__main__":
    # Quick test when run directly
    summary = get_metrics_summary()
    if summary:
        print("Metrics Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    else:
        print("Failed to collect metrics. Is the service running?")
