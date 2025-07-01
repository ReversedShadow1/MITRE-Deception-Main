'''"""
Shared metrics manager for the API.
"""

import os

from src.monitoring.metrics import MetricsManager

# Create metrics manager instance
metrics_manager = MetricsManager(
    service_name="attack_extractor",
    expose_metrics=True,
    metrics_port=int(os.environ.get("METRICS_PORT", "8001")),  # Changed to 8001 to avoid conflict with app port
    push_gateway=os.environ.get("PROMETHEUS_PUSH_GATEWAY"),
    push_interval=60
)
'''
