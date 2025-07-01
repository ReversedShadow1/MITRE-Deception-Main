# src/database/postgresql.py
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class PostgresConnection:
    """PostgreSQL database connection manager"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "postgres",
        database: str = "attack_extractor",
    ):
        """Initialize database connection parameters"""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn = None

    def connect(self) -> None:
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            logger.info(
                f"Connected to PostgreSQL: {self.host}:{self.port}/{self.database}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("PostgreSQL connection closed")

    @contextmanager
    def cursor(self):
        """Context manager for database cursor"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()

    def execute(self, query: str, *args, **kwargs) -> int:
        """Execute query and return affected row count"""
        with self.cursor() as cursor:
            cursor.execute(query, *args, **kwargs)
            return cursor.rowcount

    def query(self, query: str, *args, **kwargs) -> List[Dict[str, Any]]:
        """Execute query and return all results"""
        with self.cursor() as cursor:
            cursor.execute(query, *args, **kwargs)
            return cursor.fetchall()

    def query_one(self, query: str, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute query and return first result"""
        with self.cursor() as cursor:
            cursor.execute(query, *args, **kwargs)
            return cursor.fetchone()

    def explain_analyze(self, query: str, *args, **kwargs) -> Dict:
        """
        Run EXPLAIN ANALYZE on a query to understand performance

        Args:
            query: The SQL query to analyze

        Returns:
            Query plan as a dictionary
        """
        with self.cursor() as cursor:
            explain_query = f"EXPLAIN (FORMAT JSON, ANALYZE, VERBOSE, BUFFERS) {query}"
            cursor.execute(explain_query, *args, **kwargs)
            plan = cursor.fetchone()[0]
            return plan

    def create_optimization_report(self) -> Dict:
        """
        Create a report of potentially problematic queries and tables

        Returns:
            Dictionary with optimization suggestions
        """
        report = {
            "missing_indexes": [],
            "slow_queries": [],
            "table_bloat": [],
            "index_bloat": [],
        }

        # Check for missing indexes on foreign keys
        with self.cursor() as cursor:
            cursor.execute(
                """
            SELECT
                c.conrelid::regclass AS table_name,
                a.attname AS column_name,
                c.conname AS constraint_name
            FROM
                pg_constraint c
                JOIN pg_attribute a ON a.attnum = ANY(c.conkey) AND a.attrelid = c.conrelid
                LEFT JOIN pg_index i ON i.indrelid = c.conrelid AND a.attnum = ANY(i.indkey)
            WHERE
                c.contype = 'f'
                AND i.indexrelid IS NULL
            ORDER BY
                table_name, column_name;
            """
            )

            for row in cursor:
                report["missing_indexes"].append(
                    {
                        "table": row["table_name"],
                        "column": row["column_name"],
                        "constraint": row["constraint_name"],
                    }
                )

        # Check for slow queries
        with self.cursor() as cursor:
            cursor.execute(
                """
            SELECT
                round(total_exec_time::numeric, 2) as total_exec_time,
                calls,
                round(mean_exec_time::numeric, 2) as mean_exec_time,
                round((100 * total_exec_time / sum(total_exec_time) OVER())::numeric, 2) as percentage_cpu,
                regexp_replace(query, '\\s+', ' ', 'g') as query
            FROM
                pg_stat_statements
            ORDER BY
                total_exec_time DESC
            LIMIT 10;
            """
            )

            for row in cursor:
                report["slow_queries"].append(dict(row))

        # Check for table bloat
        with self.cursor() as cursor:
            cursor.execute(
                """
            SELECT
                schemaname || '.' || tablename as table_name,
                pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
                pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) as table_size,
                pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename) - pg_relation_size(schemaname || '.' || tablename)) as index_size,
                CASE WHEN reltuples = 0 THEN 0 ELSE round(100 * (relpages::float / (reltuples / (bs::float / (tpl_data_size + tpl_hdr_size + array_header_size + 1)))) - 100) END AS bloat_estimate
            FROM (
                SELECT
                    nn.nspname as schemaname,
                    cc.relname as tablename,
                    cc.reltuples,
                    cc.relpages,
                    bs,
                    24 AS tpl_hdr_size,
                    CASE WHEN max(coalesce(s.null_frac,0)) > 0 THEN 2 ELSE 0 END AS array_header_size,
                    sum((1-coalesce(s.null_frac, 0)) * coalesce(s.avg_width, 1024)) AS tpl_data_size
                FROM
                    pg_class cc
                    JOIN pg_namespace nn ON cc.relnamespace = nn.oid
                    JOIN pg_statistic s ON s.starelid = cc.oid
                    CROSS JOIN (SELECT current_setting('block_size')::integer AS bs) bs
                WHERE
                    cc.relkind = 'r'
                GROUP BY 1,2,3,4,5,6
            ) bloat_calculation
            ORDER BY bloat_estimate DESC
            LIMIT 10;
            """
            )

            for row in cursor:
                report["table_bloat"].append(dict(row))

        return report

    def optimize_query(self, query: str, *args, **kwargs) -> Tuple[str, Dict]:
        """
        Analyze and optimize a query

        Args:
            query: The SQL query to optimize

        Returns:
            Tuple of optimized query and optimization notes
        """
        notes = {"original_query": query}

        # Get the query plan
        plan = self.explain_analyze(query, *args, **kwargs)
        notes["original_plan"] = plan

        # Extract planning time and execution time
        planning_time = plan[0].get("Planning Time", 0)
        execution_time = plan[0].get("Execution Time", 0)
        notes["original_timing"] = {
            "planning_time_ms": planning_time,
            "execution_time_ms": execution_time,
            "total_time_ms": planning_time + execution_time,
        }

        # Analyze the query plan for common issues
        optimized_query = query
        optimizations = []

        # Check for sequential scans on large tables
        sequential_scans = self._find_sequential_scans(plan[0])
        if sequential_scans:
            for scan in sequential_scans:
                table_name = scan.get("Relation Name")
                if table_name:
                    # Check table size
                    with self.cursor() as cursor:
                        cursor.execute(
                            f"SELECT pg_size_pretty(pg_total_relation_size('{table_name}'))"
                        )
                        table_size = cursor.fetchone()[0]

                        if table_size.endswith("MB") or table_size.endswith("GB"):
                            # Large table with sequential scan, suggest index
                            filter_cond = scan.get("Filter")
                            if filter_cond:
                                # Extract column names from filter condition
                                columns = self._extract_columns_from_filter(filter_cond)
                                if columns:
                                    optimizations.append(
                                        f"Consider adding index on {table_name}({', '.join(columns)})"
                                    )

        # Suggest query rewriting if beneficial
        if (
            "Hash Join" in str(plan)
            and "Nested Loop" not in str(plan)
            and execution_time > 1000
        ):
            optimizations.append(
                "Consider rewriting to use nested loop joins for small result sets"
            )

        if "Sort" in str(plan) and execution_time > 1000:
            optimizations.append("Consider adding an index to avoid sorting")

        if len(optimizations) > 0:
            notes["optimizations"] = optimizations
        else:
            notes["optimizations"] = ["No obvious optimizations found"]

        return optimized_query, notes

    def _find_sequential_scans(self, plan_node: Dict) -> List[Dict]:
        """
        Recursively find sequential scans in a query plan

        Args:
            plan_node: A node in the query plan

        Returns:
            List of sequential scan nodes
        """
        sequential_scans = []

        node_type = plan_node.get("Node Type")
        if node_type == "Seq Scan":
            sequential_scans.append(plan_node)

        # Recursively check child plans
        for child_key in ["Plans", "Plan"]:
            if child_key in plan_node:
                child_plans = plan_node[child_key]
                if isinstance(child_plans, list):
                    for child_plan in child_plans:
                        sequential_scans.extend(self._find_sequential_scans(child_plan))
                elif isinstance(child_plans, dict):
                    sequential_scans.extend(self._find_sequential_scans(child_plans))

        return sequential_scans

    def _extract_columns_from_filter(self, filter_str: str) -> List[str]:
        """
        Extract column names from a filter condition

        Args:
            filter_str: SQL filter condition

        Returns:
            List of column names
        """
        columns = []

        # This is a simplified implementation - a real one would need more robust parsing
        parts = filter_str.split()
        for part in parts:
            # Remove punctuation
            clean_part = part.strip("()=><!'\"")
            # If it looks like an identifier and not a number or function
            if (
                clean_part
                and clean_part[0].isalpha()
                and not clean_part.isdigit()
                and "(" not in clean_part
            ):
                if clean_part not in [
                    "AND",
                    "OR",
                    "NOT",
                    "IS",
                    "NULL",
                    "IN",
                    "LIKE",
                    "BETWEEN",
                ]:
                    columns.append(clean_part)

        return columns


# Singleton database connection
_db_connection = None


def get_db() -> PostgresConnection:
    """Get database connection (singleton)"""
    global _db_connection

    if _db_connection is None:
        _db_connection = PostgresConnection(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            database=os.environ.get("POSTGRES_DB", "attack_extractor"),
        )
        _db_connection.connect()

    return _db_connection
