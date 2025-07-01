import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.monitoring.metrics import MetricsManager


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for request monitoring and metrics collection"""

    def __init__(self, app: FastAPI, metrics_manager: MetricsManager):
        """
        Initialize middleware

        Args:
            app: FastAPI application
            metrics_manager: Metrics manager instance
        """
        super().__init__(app)
        self.metrics_manager = metrics_manager

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process request and collect metrics

        Args:
            request: HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response
        """
        # Record start time
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Record exception
            status_code = 500
            raise e
        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Extract method and endpoint
            method = request.method
            endpoint = request.url.path

            # Track request metrics
            self.metrics_manager.track_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration,
            )

        return response


class DatabaseMonitoringMiddleware:
    """Middleware for database query monitoring"""

    def __init__(self, metrics_manager: MetricsManager):
        """
        Initialize middleware

        Args:
            metrics_manager: Metrics manager instance
        """
        self.metrics_manager = metrics_manager

    def wrap_neo4j(self, neo4j_connector):
        """
        Wrap Neo4j connector to monitor queries

        Args:
            neo4j_connector: Neo4j connector instance

        Returns:
            Wrapped Neo4j connector
        """
        # Check which method exists on the connector
        if hasattr(neo4j_connector, "run_query"):
            # This is the Neo4jConnector from enhanced_attack_extractor.py
            original_method = neo4j_connector.run_query
            method_name = "run_query"
        elif hasattr(neo4j_connector, "run"):
            # This is the Neo4jConnection from database/neo4j.py
            original_method = neo4j_connector.run
            method_name = "run"
        else:
            # Unknown connector type, skip monitoring
            import logging

            logging.warning(
                "Neo4j connector has neither 'run' nor 'run_query' method, skipping monitoring"
            )
            return neo4j_connector

        def wrapped_method(query, params=None):
            # Record start time
            start_time = time.time()

            try:
                # Execute query
                result = original_method(query, params)
                return result
            except Exception as e:
                # Record error
                self.metrics_manager.track_database_error(database="neo4j")
                raise e
            finally:
                # Calculate duration and track metrics
                duration = time.time() - start_time

                # Determine query type
                query_type = self._determine_query_type(query)

                # Track query metrics
                self.metrics_manager.track_database_query(
                    database="neo4j", query_type=query_type, duration=duration
                )

        # Replace original method
        setattr(neo4j_connector, method_name, wrapped_method)

        return neo4j_connector

    def wrap_postgres(self, postgres_connector):
        """
        Wrap PostgreSQL connector to monitor queries

        Args:
            postgres_connector: PostgreSQL connector instance

        Returns:
            Wrapped PostgreSQL connector
        """
        # Check if the connector has the expected methods
        if not hasattr(postgres_connector, "execute"):
            import logging

            logging.warning(
                "PostgreSQL connector missing 'execute' method, skipping monitoring"
            )
            return postgres_connector

        original_execute = postgres_connector.execute

        # These methods might not exist on all connectors, so check first
        original_query = getattr(postgres_connector, "query", None)
        original_query_one = getattr(postgres_connector, "query_one", None)

        def wrapped_execute(query, *args, **kwargs):
            # Record start time
            start_time = time.time()

            try:
                # Execute query
                result = original_execute(query, *args, **kwargs)
                return result
            except Exception as e:
                # Record error
                self.metrics_manager.track_database_error(database="postgres")
                raise e
            finally:
                # Calculate duration and track metrics
                duration = time.time() - start_time

                # Determine query type
                query_type = self._determine_query_type(query)

                # Track query metrics
                self.metrics_manager.track_database_query(
                    database="postgres", query_type=query_type, duration=duration
                )

        def wrapped_query(query, *args, **kwargs):
            # Record start time
            start_time = time.time()

            try:
                # Execute query
                result = original_query(query, *args, **kwargs)
                return result
            except Exception as e:
                # Record error
                self.metrics_manager.track_database_error(database="postgres")
                raise e
            finally:
                # Calculate duration and track metrics
                duration = time.time() - start_time

                # Determine query type
                query_type = self._determine_query_type(query)

                # Track query metrics
                self.metrics_manager.track_database_query(
                    database="postgres", query_type=query_type, duration=duration
                )

        def wrapped_query_one(query, *args, **kwargs):
            # Record start time
            start_time = time.time()

            try:
                # Execute query
                result = original_query_one(query, *args, **kwargs)
                return result
            except Exception as e:
                # Record error
                self.metrics_manager.track_database_error(database="postgres")
                raise e
            finally:
                # Calculate duration and track metrics
                duration = time.time() - start_time

                # Determine query type
                query_type = self._determine_query_type(query)

                # Track query metrics
                self.metrics_manager.track_database_query(
                    database="postgres", query_type=query_type, duration=duration
                )

        # Replace original methods
        postgres_connector.execute = wrapped_execute

        if original_query:
            postgres_connector.query = wrapped_query

        if original_query_one:
            postgres_connector.query_one = wrapped_query_one

        return postgres_connector

    def _determine_query_type(self, query: str) -> str:
        """
        Determine query type from query string

        Args:
            query: SQL or Cypher query

        Returns:
            Query type
        """
        if not query or not isinstance(query, str):
            return "unknown"

        query = query.strip().upper()

        if query.startswith("SELECT"):
            return "select"
        elif query.startswith("INSERT"):
            return "insert"
        elif query.startswith("UPDATE"):
            return "update"
        elif query.startswith("DELETE"):
            return "delete"
        elif query.startswith("CREATE"):
            return "create"
        elif query.startswith("DROP"):
            return "drop"
        elif query.startswith("ALTER"):
            return "alter"
        elif query.startswith("MATCH") and "CREATE" in query:
            return "match_create"
        elif query.startswith("MATCH") and "DELETE" in query:
            return "match_delete"
        elif query.startswith("MATCH") and "SET" in query:
            return "match_update"
        elif query.startswith("MATCH"):
            return "match"
        elif query.startswith("MERGE"):
            return "merge"
        else:
            return "other"
