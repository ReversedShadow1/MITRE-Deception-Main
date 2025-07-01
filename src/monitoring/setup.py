import os
from typing import Dict, Optional

from fastapi import FastAPI

from src.monitoring.logging_setup import setup_logging
from src.monitoring.metrics import MetricsManager
from src.monitoring.middleware import DatabaseMonitoringMiddleware, MonitoringMiddleware


def setup_monitoring(
    app: FastAPI,
    service_name: str = "attack_extractor",
    log_level: str = "INFO",
    log_dir: str = "logs",
    elasticsearch_host: Optional[str] = None,
    metrics_port: int = 8000,
    prometheus_push_gateway: Optional[str] = None,
) -> Dict:
    """
    Setup monitoring, logging, and metrics collection

    Args:
        app: FastAPI application
        service_name: Service name for identification
        log_level: Logging level
        log_dir: Directory for log files
        elasticsearch_host: Elasticsearch host (if None, Elasticsearch logging is disabled)
        metrics_port: Port for Prometheus metrics
        prometheus_push_gateway: Prometheus push gateway URL

    Returns:
        Dictionary with monitoring components
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(log_dir, f"{service_name}.json")
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        elasticsearch_host=elasticsearch_host,
        service_name=service_name,
    )

    # Setup metrics
    metrics_manager = MetricsManager(
        service_name=service_name,
        expose_metrics=True,
        metrics_port=metrics_port,
        push_gateway=prometheus_push_gateway,
    )

    # Add monitoring middleware to app
    app.add_middleware(MonitoringMiddleware, metrics_manager=metrics_manager)

    # Create database monitoring middleware
    db_monitoring = DatabaseMonitoringMiddleware(metrics_manager)

    return {
        "metrics_manager": metrics_manager,
        "db_monitoring": db_monitoring,
    }


def wrap_database_connections(monitoring_components: Dict):
    """
    Wrap database connections with monitoring middleware

    Args:
        monitoring_components: Dictionary of monitoring components
    """
    from src.database.neo4j import get_neo4j
    from src.database.postgresql import get_db

    # Get database connections
    neo4j_conn = get_neo4j()
    postgres_conn = get_db()

    # Wrap with monitoring middleware
    db_monitoring = monitoring_components["db_monitoring"]
    db_monitoring.wrap_neo4j(neo4j_conn)
    db_monitoring.wrap_postgres(postgres_conn)
