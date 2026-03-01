"""
Metrics collection for monitoring and dashboards.
In-memory metrics with rolling windows.
"""
import time
import logging
from collections import deque
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    """Single request metric."""
    timestamp: float
    latency_ms: float
    tokens: int
    success: bool


class MetricsCollector:
    """
    In-memory metrics collector with rolling window.
    Thread-safe for concurrent access.
    """

    def __init__(self, window_size: int = 1000, window_duration_seconds: int = 3600):
        """
        Initialize metrics collector.

        Args:
            window_size: Maximum number of requests to track
            window_duration_seconds: Time window for metrics (1 hour default)
        """
        self.window_size = window_size
        self.window_duration = window_duration_seconds
        self.requests: deque = deque(maxlen=window_size)
        self.lock = Lock()

        # Counters
        self.total_requests = 0
        self.total_errors = 0
        self.total_tokens = 0

        # Start time
        self.start_time = time.time()

        logger.info(f"[METRICS] Initialized (window: {window_size} requests, {window_duration_seconds}s)")

    def record_request(
        self,
        latency_ms: float,
        tokens: int = 0,
        success: bool = True
    ):
        """
        Record a request metric.

        Args:
            latency_ms: Request latency in milliseconds
            tokens: Number of tokens in response
            success: Whether request succeeded
        """
        metric = RequestMetric(
            timestamp=time.time(),
            latency_ms=latency_ms,
            tokens=tokens,
            success=success
        )

        with self.lock:
            self.requests.append(metric)
            self.total_requests += 1
            self.total_tokens += tokens

            if not success:
                self.total_errors += 1

    def _get_recent_requests(self) -> list:
        """Get requests within the time window."""
        cutoff = time.time() - self.window_duration

        with self.lock:
            return [r for r in self.requests if r.timestamp > cutoff]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary for dashboard.

        Returns:
            Dict with latency, throughput, and error metrics
        """
        recent = self._get_recent_requests()

        if not recent:
            return {
                "uptime_seconds": int(time.time() - self.start_time),
                "total_requests": self.total_requests,
                "window_requests": 0,
                "latency": {
                    "avg_ms": 0,
                    "p50_ms": 0,
                    "p95_ms": 0,
                    "p99_ms": 0
                },
                "throughput": {
                    "requests_per_minute": 0,
                    "tokens_per_minute": 0
                },
                "errors": {
                    "total": self.total_errors,
                    "rate_percent": 0
                }
            }

        # Calculate latency percentiles
        latencies = sorted([r.latency_ms for r in recent])
        n = len(latencies)

        def percentile(data, p):
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        # Calculate throughput
        window_seconds = min(
            time.time() - self.start_time,
            self.window_duration
        )
        window_minutes = max(window_seconds / 60, 1)

        recent_tokens = sum(r.tokens for r in recent)
        recent_errors = sum(1 for r in recent if not r.success)

        return {
            "uptime_seconds": int(time.time() - self.start_time),
            "total_requests": self.total_requests,
            "window_requests": n,
            "latency": {
                "avg_ms": round(sum(latencies) / n, 2),
                "p50_ms": round(percentile(latencies, 50), 2),
                "p95_ms": round(percentile(latencies, 95), 2),
                "p99_ms": round(percentile(latencies, 99), 2),
                "min_ms": round(min(latencies), 2),
                "max_ms": round(max(latencies), 2)
            },
            "throughput": {
                "requests_per_minute": round(n / window_minutes, 2),
                "tokens_per_minute": round(recent_tokens / window_minutes, 2)
            },
            "errors": {
                "total": self.total_errors,
                "window_errors": recent_errors,
                "rate_percent": round(recent_errors / n * 100, 2) if n > 0 else 0
            }
        }

    def get_latency_histogram(self, buckets: list = None) -> Dict[str, int]:
        """
        Get latency distribution histogram.

        Args:
            buckets: Latency bucket boundaries in ms

        Returns:
            Dict mapping bucket labels to counts
        """
        if buckets is None:
            buckets = [100, 500, 1000, 2000, 5000, 10000]

        recent = self._get_recent_requests()
        latencies = [r.latency_ms for r in recent]

        histogram = {}
        prev = 0

        for bucket in buckets:
            label = f"{prev}-{bucket}ms"
            histogram[label] = sum(1 for l in latencies if prev <= l < bucket)
            prev = bucket

        # Add overflow bucket
        histogram[f">{buckets[-1]}ms"] = sum(1 for l in latencies if l >= buckets[-1])

        return histogram

    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.requests.clear()
            self.total_requests = 0
            self.total_errors = 0
            self.total_tokens = 0
            self.start_time = time.time()

        logger.info("[METRICS] Reset complete")
