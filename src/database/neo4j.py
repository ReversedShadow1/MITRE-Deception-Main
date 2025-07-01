# src/database/neo4j.py
import logging
import os
import time
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase, Session

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Neo4j database connection manager"""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "pipe",
    ):
        """Initialize Neo4j connection parameters"""
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

    def connect(self) -> bool:
        """Connect to the Neo4j database.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # Test connection with retry logic
            max_retries = 3
            retry_delay = 2  # seconds

            for attempt in range(max_retries):
                try:
                    with self.driver.session(database=self.database) as session:
                        result = session.run(
                            "MATCH (n) RETURN count(n) AS count LIMIT 1"
                        )
                        count = result.single()["count"]
                        logger.info(
                            f"Successfully connected to Neo4j database. Node count: {count}"
                        )
                    return True
                except Exception as retry_error:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Neo4j connection attempt {attempt+1} failed. Retrying in {retry_delay}s: {retry_error}"
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise retry_error

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            return False

    def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def run(self, query: str, params: Dict = None) -> List[Dict]:
        """Run a Cypher query"""
        if not self.driver:
            logger.error("No active Neo4j connection")
            return []

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing Neo4j query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return []


# Singleton Neo4j connection
_neo4j_connection = None


def get_neo4j() -> Neo4jConnection:
    """Get Neo4j connection (singleton)"""
    global _neo4j_connection

    if _neo4j_connection is None:
        _neo4j_connection = Neo4jConnection(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
            database=os.environ.get("NEO4J_DATABASE", "pipe"),
        )
        _neo4j_connection.connect()

    return _neo4j_connection
