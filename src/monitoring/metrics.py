import time
from typing import Any, Dict, List, Optional, Tuple, Union

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    push_to_gateway,
    start_http_server,
)


class MetricsManager:
    """Manages application metrics collection and exposition"""

    def __init__(
        self,
        service_name: str = "attack_extractor",
        expose_metrics: bool = True,
        metrics_port: int = 8000,
        push_gateway: str = None,
        push_interval: int = 60,
    ):
        """
        Initialize metrics manager

        Args:
            service_name: Service name used as prefix for metrics
            expose_metrics: Whether to expose metrics HTTP endpoint
            metrics_port: Port for metrics HTTP server
            push_gateway: Prometheus push gateway URL (if None, push is disabled)
            push_interval: Interval in seconds between pushes to gateway
        """
        self.service_name = service_name
        self.expose_metrics = expose_metrics
        self.metrics_port = metrics_port
        self.push_gateway = push_gateway
        self.push_interval = push_interval

        # Setup metrics registry
        self.metrics = {}
        self._register_default_metrics()

        # Start HTTP server if enabled
        if expose_metrics:
            start_http_server(metrics_port)
            print(f"Metrics server started on port {metrics_port}")

        # Setup push gateway thread if enabled
        if push_gateway:
            import threading

            self.push_thread = threading.Thread(
                target=self._push_metrics_loop, daemon=True
            )
            self.push_thread.start()

    def _register_default_metrics(self):
        """Register default metrics for monitoring"""
        # System metrics
        self.metrics["process_start_time"] = Gauge(
            f"{self.service_name}_process_start_time_seconds",
            "Process start time in seconds since Unix epoch",
        )
        self.metrics["process_start_time"].set_to_current_time()

        # API metrics
        self.metrics["http_requests_total"] = Counter(
            f"{self.service_name}_http_requests_total",
            "Total number of HTTP requests",
            ["method", "endpoint", "status"],
        )

        self.metrics["http_request_duration_seconds"] = Histogram(
            f"{self.service_name}_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=(
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                float("inf"),
            ),
        )

        # Extraction metrics
        self.metrics["extraction_requests_total"] = Counter(
            f"{self.service_name}_extraction_requests_total",
            "Total number of technique extraction requests",
            ["method", "status"],
        )

        self.metrics["extraction_processing_seconds"] = Histogram(
            f"{self.service_name}_extraction_processing_seconds",
            "Time spent processing extraction requests",
            ["extractor_type"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf")),
        )

        self.metrics["extraction_technique_count"] = Histogram(
            f"{self.service_name}_extraction_technique_count",
            "Number of techniques identified per request",
            buckets=(0, 1, 2, 5, 10, 20, 50, 100, float("inf")),
        )

        self.metrics["extraction_confidence"] = Histogram(
            f"{self.service_name}_extraction_confidence",
            "Confidence scores of extracted techniques",
            ["extractor_type"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

        # Database metrics
        self.metrics["database_query_duration_seconds"] = Histogram(
            f"{self.service_name}_database_query_duration_seconds",
            "Database query duration in seconds",
            ["database", "query_type"],
            buckets=(
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                float("inf"),
            ),
        )

        self.metrics["database_connection_errors"] = Counter(
            f"{self.service_name}_database_connection_errors_total",
            "Total number of database connection errors",
            ["database"],
        )

        # Queue metrics
        self.metrics["queue_jobs_total"] = Gauge(
            f"{self.service_name}_queue_jobs_total",
            "Total number of jobs in queue",
            ["queue", "status"],
        )

        self.metrics["queue_processing_time_seconds"] = Histogram(
            f"{self.service_name}_queue_processing_time_seconds",
            "Job processing time in seconds",
            ["queue"],
            buckets=(
                0.1,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                30.0,
                60.0,
                120.0,
                300.0,
                float("inf"),
            ),
        )

        # Model metrics
        self.metrics["model_load_time_seconds"] = Histogram(
            f"{self.service_name}_model_load_time_seconds",
            "Time spent loading models",
            ["model_type"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float("inf")),
        )

        self.metrics["model_inference_time_seconds"] = Histogram(
            f"{self.service_name}_model_inference_time_seconds",
            "Time spent on model inference",
            ["model_type"],
            buckets=(
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                float("inf"),
            ),
        )

    def _push_metrics_loop(self):
        """Push metrics to gateway at regular intervals"""
        while True:
            try:
                # Push metrics to gateway
                push_to_gateway(
                    self.push_gateway,
                    job=self.service_name,
                    registry=None,  # Use default registry
                )
            except Exception as e:
                print(f"Error pushing metrics to gateway: {e}")

            # Sleep until next push
            time.sleep(self.push_interval)

    def track_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """
        Track HTTP request metrics

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: Request endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
        """
        status = str(status_code)

        # Track request count
        self.metrics["http_requests_total"].labels(
            method=method, endpoint=endpoint, status=status
        ).inc()

        # Track request duration
        self.metrics["http_request_duration_seconds"].labels(
            method=method, endpoint=endpoint
        ).observe(duration)

    def track_extraction(self, method: str, status: str, techniques_count: int = 0):
        """
        Track extraction request metrics

        Args:
            method: Extraction method
            status: Request status (started, success, error)
            techniques_count: Number of techniques extracted
        """
        # Track request count
        self.metrics["extraction_requests_total"].labels(
            method=method, status=status
        ).inc()

        # Track techniques count if successful
        if status == "success" and techniques_count > 0:
            self.metrics["extraction_technique_count"].observe(techniques_count)

    def track_extraction_time(self, extractor_type: str, duration: float):
        """
        Track extraction processing time

        Args:
            extractor_type: Type of extractor
            duration: Processing time in seconds
        """
        self.metrics["extraction_processing_seconds"].labels(
            extractor_type=extractor_type
        ).observe(duration)

    def track_technique_confidence(self, extractor_type: str, confidence: float):
        """
        Track technique confidence score

        Args:
            extractor_type: Type of extractor
            confidence: Confidence score (0-1)
        """
        self.metrics["extraction_confidence"].labels(
            extractor_type=extractor_type
        ).observe(confidence)

    def track_database_query(self, database: str, query_type: str, duration: float):
        """
        Track database query performance

        Args:
            database: Database name
            query_type: Type of query
            duration: Query duration in seconds
        """
        self.metrics["database_query_duration_seconds"].labels(
            database=database, query_type=query_type
        ).observe(duration)

    def track_database_error(self, database: str):
        """
        Track database connection error

        Args:
            database: Database name
        """
        self.metrics["database_connection_errors"].labels(database=database).inc()

    def update_queue_stats(self, queue: str, status: str, count: int):
        """
        Update queue statistics

        Args:
            queue: Queue name
            status: Job status
            count: Number of jobs
        """
        self.metrics["queue_jobs_total"].labels(queue=queue, status=status).set(count)

    def track_queue_processing(self, queue: str, duration: float):
        """
        Track job processing time

        Args:
            queue: Queue name
            duration: Processing time in seconds
        """
        self.metrics["queue_processing_time_seconds"].labels(queue=queue).observe(
            duration
        )

    def track_model_load(self, model_type: str, duration: float):
        """
        Track model loading time

        Args:
            model_type: Type of model
            duration: Loading time in seconds
        """
        self.metrics["model_load_time_seconds"].labels(model_type=model_type).observe(
            duration
        )

    def track_model_inference(self, model_type: str, duration: float):
        """
        Track model inference time

        Args:
            model_type: Type of model
            duration: Inference time in seconds
        """
        self.metrics["model_inference_time_seconds"].labels(
            model_type=model_type
        ).observe(duration)
