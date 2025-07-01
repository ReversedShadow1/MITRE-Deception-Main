import json
import logging
import os
import socket
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

import elasticsearch
from elasticsearch import Elasticsearch, helpers
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)

        # Add timestamp in ISO format
        log_record["timestamp"] = datetime.utcnow().isoformat()

        # Add hostname
        log_record["hostname"] = socket.gethostname()

        # Add service name
        log_record["service"] = os.environ.get("SERVICE_NAME", "attack_extractor")

        # Add environment
        log_record["environment"] = os.environ.get("ENVIRONMENT", "development")

        # Add process and thread IDs
        log_record["process"] = record.process
        log_record["thread"] = record.thread


class ElasticsearchLogHandler(logging.Handler):
    """Logging handler that sends logs to Elasticsearch"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_prefix: str = "logs",
        auth=None,
        use_ssl: bool = False,
        buffer_size: int = 100,
        flush_interval: int = 5,
        **kwargs,
    ):
        super(ElasticsearchLogHandler, self).__init__()

        # Connection settings
        self.host = host
        self.port = port
        self.index_prefix = index_prefix
        self.auth = auth
        self.use_ssl = use_ssl
        self.kwargs = kwargs

        # Buffering settings
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer = []
        self.last_flush = time.time()

        # Connect to Elasticsearch
        self._connect()

        # Set formatter
        self.setFormatter(CustomJsonFormatter())

    def _connect(self):
        """Connect to Elasticsearch"""
        try:
            self.client = Elasticsearch(
                [{"host": self.host, "port": self.port}],
                http_auth=self.auth,
                use_ssl=self.use_ssl,
                **self.kwargs,
            )
            self.client.info()  # Test connection
            logging.info(f"Connected to Elasticsearch at {self.host}:{self.port}")
        except Exception as e:
            logging.error(f"Failed to connect to Elasticsearch: {e}")
            self.client = None

    def _get_index_name(self):
        """Get index name with date suffix"""
        return f"{self.index_prefix}-{datetime.utcnow().strftime('%Y.%m.%d')}"

    def emit(self, record):
        """Process log record and send to Elasticsearch"""
        if not self.client:
            return

        try:
            # Format the record
            message = self.format(record)

            # Parse JSON message
            log_entry = json.loads(message)

            # Add to buffer
            self.buffer.append({"_index": self._get_index_name(), "_source": log_entry})

            # Check if buffer should be flushed
            if (
                len(self.buffer) >= self.buffer_size
                or time.time() - self.last_flush >= self.flush_interval
            ):
                self.flush()

        except Exception as e:
            logging.error(f"Error sending log to Elasticsearch: {e}")

    def flush(self):
        """Flush buffered logs to Elasticsearch"""
        if not self.client or not self.buffer:
            return

        try:
            # Bulk send logs
            helpers.bulk(self.client, self.buffer)

            # Clear buffer and update flush time
            self.buffer = []
            self.last_flush = time.time()
        except Exception as e:
            logging.error(f"Error flushing logs to Elasticsearch: {e}")

    def close(self):
        """Close handler and flush any remaining logs"""
        self.flush()
        super(ElasticsearchLogHandler, self).close()


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    elasticsearch_host: str = None,
    elasticsearch_port: int = 9200,
    service_name: str = "attack_extractor",
):
    """
    Setup logging configuration with console, file, and Elasticsearch handlers

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, file logging is disabled)
        elasticsearch_host: Elasticsearch host (if None, Elasticsearch logging is disabled)
        elasticsearch_port: Elasticsearch port
        service_name: Service name for log identification
    """
    # Set service name in environment
    os.environ["SERVICE_NAME"] = service_name

    # Get root logger
    root_logger = logging.getLogger()

    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with plain formatter
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Create file handler with JSON formatter if log file specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_formatter = CustomJsonFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Create Elasticsearch handler if host specified
    if elasticsearch_host:
        try:
            es_handler = ElasticsearchLogHandler(
                host=elasticsearch_host,
                port=elasticsearch_port,
                index_prefix=f"{service_name}-logs",
                buffer_size=100,
                flush_interval=5,
            )
            root_logger.addHandler(es_handler)
        except Exception as e:
            logging.error(f"Failed to set up Elasticsearch logging: {e}")

    logging.info(f"Logging setup complete for service {service_name}")
