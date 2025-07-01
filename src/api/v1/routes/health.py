"""
API endpoints for health checks and system status (v1).
"""

import os
import time
from typing import Dict, List, Optional

import torch
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from src.database.neo4j import get_neo4j
from src.database.postgresql import get_db
from src.enhanced_attack_extractor import get_extractor

# Create router
router = APIRouter(tags=["health"])


# Models
class ExtractorStatus(BaseModel):
    """Status of an extractor"""

    name: str
    loaded: bool
    available: bool


class DatabaseStatus(BaseModel):
    """Status of a database connection"""

    type: str
    connected: bool
    version: Optional[str] = None
    latency_ms: float


class HealthStatus(BaseModel):
    """Health status response model"""

    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="API version")
    extractors: List[ExtractorStatus] = Field(..., description="Status of extractors")
    databases: List[DatabaseStatus] = Field(..., description="Database connections")
    environment: str = Field(..., description="Deployment environment")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    extractor_error: Optional[str] = Field(
        None, description="Extractor initialization error if any"
    )


# Global start time for uptime calculation
START_TIME = time.time()


# Endpoints
@router.get("/health", response_model=HealthStatus)
async def health_check(request: Request):
    """
    Check system health

    This endpoint provides information about the health of the system, including database
    connections, extractor availability, and resource availability.
    """
    # Try to get extractor instance - this might fail if data isn't loaded yet
    extractor = None
    extractor_available = False
    extractor_error = None

    try:
        extractor = get_extractor()
        extractor_available = True
    except Exception as e:
        extractor_available = False
        extractor_error = str(e)
        import logging

        logging.warning(f"Extractor not available: {str(e)}")

    # Check extractor status
    extractor_statuses = []
    if extractor_available and extractor is not None:
        # Extractor is available, check individual extractors
        for name in ["rule_based", "bm25", "ner", "semantic", "classifier", "kev"]:
            try:
                is_loaded = (
                    hasattr(extractor, "extractors")
                    and name in extractor.extractors
                    and extractor.extractors[name] is not None
                )
                extractor_statuses.append(
                    {
                        "name": name,
                        "loaded": is_loaded,
                        "available": True,
                    }
                )
            except Exception as e:
                extractor_statuses.append(
                    {
                        "name": name,
                        "loaded": False,
                        "available": False,
                    }
                )
    else:
        # Extractor not available, mark all as unavailable
        for name in ["rule_based", "bm25", "ner", "semantic", "classifier", "kev"]:
            extractor_statuses.append(
                {
                    "name": name,
                    "loaded": False,
                    "available": False,
                }
            )

    # Check database connections
    database_statuses = []

    # Neo4j status
    neo4j = get_neo4j()
    neo4j_start = time.time()
    neo4j_connected = False
    neo4j_version = None

    try:
        # Try simple query to check connection
        result = neo4j.run("RETURN 1 as test")
        neo4j_connected = len(result) > 0

        # Get Neo4j version
        try:
            version_result = neo4j.run(
                "CALL dbms.components() YIELD name, versions RETURN name, versions"
            )
            if version_result and len(version_result) > 0:
                neo4j_version = f"{version_result[0].get('name')} {version_result[0].get('versions')[0]}"
        except Exception:
            # If version query fails, just continue without version info
            pass

    except Exception as e:
        import logging

        logging.warning(f"Neo4j health check failed: {str(e)}")

    neo4j_latency = (time.time() - neo4j_start) * 1000  # Convert to milliseconds

    database_statuses.append(
        {
            "type": "neo4j",
            "connected": neo4j_connected,
            "version": neo4j_version,
            "latency_ms": neo4j_latency,
        }
    )

    # PostgreSQL status
    postgres = get_db()
    postgres_start = time.time()
    postgres_connected = False
    postgres_version = None

    try:
        # Try simple query to check connection
        result = postgres.query_one("SELECT version();")
        postgres_connected = result is not None

        # Get PostgreSQL version
        if result and "version" in result:
            postgres_version = result["version"].split(",")[0]
    except Exception as e:
        import logging

        logging.warning(f"PostgreSQL health check failed: {str(e)}")

    postgres_latency = (time.time() - postgres_start) * 1000  # Convert to milliseconds

    database_statuses.append(
        {
            "type": "postgresql",
            "connected": postgres_connected,
            "version": postgres_version,
            "latency_ms": postgres_latency,
        }
    )

    # Redis status if we're using it
    try:
        from src.queue.manager import AnalysisQueueManager

        redis_start = time.time()
        queue_manager = AnalysisQueueManager()
        redis_connected = queue_manager.redis_conn.ping()
        redis_info = queue_manager.redis_conn.info()
        redis_version = redis_info.get("redis_version")
        redis_latency = (time.time() - redis_start) * 1000

        database_statuses.append(
            {
                "type": "redis",
                "connected": redis_connected,
                "version": redis_version,
                "latency_ms": redis_latency,
            }
        )
    except Exception as e:
        import logging

        logging.warning(f"Redis health check failed: {str(e)}")

    # Determine overall status
    overall_status = "healthy"
    if not neo4j_connected or not postgres_connected:
        overall_status = "degraded"

    # Check extractor availability
    if not extractor_available:
        overall_status = "degraded"

    # Get environment
    environment = os.environ.get("ENVIRONMENT", "development")

    # Check GPU availability
    gpu_available = torch.cuda.is_available()

    # Calculate uptime
    uptime_seconds = time.time() - START_TIME

    response = {
        "status": overall_status,
        "version": "1.0.0",
        "extractors": extractor_statuses,
        "databases": database_statuses,
        "environment": environment,
        "gpu_available": gpu_available,
        "uptime_seconds": uptime_seconds,
    }

    # Add extractor error if present
    if extractor_error:
        response["extractor_error"] = extractor_error

    return response
