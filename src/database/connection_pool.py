import logging
import time
from threading import Lock
from typing import Any, Dict, Optional

from psycopg2 import pool

logger = logging.getLogger(__name__)


class PostgresConnectionPool:
    """Thread-safe PostgreSQL connection pool"""

    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls, *args, **kwargs):
        """Singleton pattern implementation"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(*args, **kwargs)
            return cls._instance

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "postgres",
        database: str = "attack_extractor",
        min_connections: int = 5,
        max_connections: int = 20,
    ):
        """Initialize the connection pool"""
        self.connection_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
        }
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool = None
        self._create_pool()

        # Performance metrics
        self.connection_wait_time = 0
        self.connection_request_count = 0
        self.connection_errors = 0

    def _create_pool(self) -> None:
        """Create the connection pool"""
        try:
            self._pool = pool.ThreadedConnectionPool(
                self.min_connections, self.max_connections, **self.connection_params
            )
            logger.info(
                f"Created PostgreSQL connection pool with {self.min_connections}-{self.max_connections} connections"
            )
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            self.connection_errors += 1
            raise

    def get_connection(self):
        """Get a connection from the pool with metrics collection"""
        if not self._pool:
            self._create_pool()

        start_time = time.time()
        self.connection_request_count += 1

        try:
            connection = self._pool.getconn()
            self.connection_wait_time += time.time() - start_time
            return connection
        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}")
            self.connection_errors += 1
            raise

    def return_connection(self, connection) -> None:
        """Return a connection to the pool"""
        if self._pool:
            try:
                self._pool.putconn(connection)
            except Exception as e:
                logger.error(f"Error returning connection to pool: {e}")
                self.connection_errors += 1

    def close_all(self) -> None:
        """Close all connections in the pool"""
        if self._pool:
            self._pool.closeall()
            logger.info("Closed all connections in the pool")

    def get_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics"""
        avg_wait_time = 0
        if self.connection_request_count > 0:
            avg_wait_time = self.connection_wait_time / self.connection_request_count

        return {
            "pool_size": {
                "min": self.min_connections,
                "max": self.max_connections,
                "current": self._pool._used + self._pool._idle if self._pool else 0,
            },
            "connections": {
                "used": self._pool._used if self._pool else 0,
                "idle": self._pool._idle if self._pool else 0,
                "total_requests": self.connection_request_count,
                "errors": self.connection_errors,
            },
            "performance": {
                "avg_wait_time_ms": avg_wait_time * 1000,
                "total_wait_time_ms": self.connection_wait_time * 1000,
            },
        }


# Update the get_db function to use connection pooling
def get_pooled_db():
    """Get a database connection from the pool"""
    # Get connection parameters from environment
    import os

    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    database = os.environ.get("POSTGRES_DB", "attack_extractor")

    # Get connection pool
    pool = PostgresConnectionPool.get_instance(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        min_connections=5,
        max_connections=20,
    )

    connection = pool.get_connection()

    # Create a wrapper with the same interface as PostgresConnection
    from src.database.postgresql import PostgresConnection

    db = PostgresConnection._from_connection(connection, pool)

    return db
