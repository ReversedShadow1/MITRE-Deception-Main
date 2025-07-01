import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg2
import schedule
from neo4j import GraphDatabase
from psycopg2.extras import Json, RealDictCursor

from src.database.neo4j import get_neo4j
from src.database.postgresql import get_db

logger = logging.getLogger(__name__)


class DatabaseSyncManager:
    """
    Manages synchronization between Neo4j and PostgreSQL databases
    """

    def __init__(self):
        """Initialize the sync manager"""
        self.neo4j = get_neo4j()
        self.postgres = get_db()

        # Initialize sync schema if needed
        self._initialize_sync_schema()

    def _initialize_sync_schema(self):
        """Initialize database schema for sync tracking"""
        # Check if sync_metadata table exists
        result = self.postgres.query_one(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'sync_metadata'
            ) as exists
            """
        )

        if not result or not result.get("exists", False):
            # Create sync tracking tables
            self.postgres.execute(
                """
                CREATE TABLE sync_metadata (
                    id SERIAL PRIMARY KEY,
                    entity_type VARCHAR(50) NOT NULL,
                    last_sync_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    entity_count INTEGER NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            self.postgres.execute(
                """
                CREATE TABLE sync_conflicts (
                    id SERIAL PRIMARY KEY,
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id VARCHAR(255) NOT NULL,
                    source_db VARCHAR(20) NOT NULL,
                    conflict_type VARCHAR(50) NOT NULL,
                    resolution VARCHAR(50),
                    source_data JSONB,
                    target_data JSONB,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP WITH TIME ZONE
                )
                """
            )

            self.postgres.execute(
                """
                CREATE INDEX idx_sync_conflicts_entity_type ON sync_conflicts(entity_type);
                CREATE INDEX idx_sync_conflicts_entity_id ON sync_conflicts(entity_id);
                CREATE INDEX idx_sync_conflicts_resolved ON sync_conflicts(resolved);
                """
            )

            logger.info("Created sync tracking schema")

    def sync_all(self):
        """
        Synchronize all entity types between databases
        """
        entity_types = ["technique", "tactic", "mitigation", "group", "software"]

        for entity_type in entity_types:
            self.sync_entity_type(entity_type)

    def sync_entity_type(self, entity_type: str):
        """
        Synchronize a specific entity type between databases

        Args:
            entity_type: Type of entity to synchronize
        """
        logger.info(f"Starting synchronization for entity type: {entity_type}")

        try:
            # Get sync direction - which database is source of truth for this entity type
            sync_direction = self._get_sync_direction(entity_type)

            if sync_direction == "neo4j_to_postgres":
                self._sync_from_neo4j_to_postgres(entity_type)
            elif sync_direction == "postgres_to_neo4j":
                self._sync_from_postgres_to_neo4j(entity_type)
            else:
                logger.error(f"Unknown sync direction: {sync_direction}")
                return

            # Record successful sync
            self._record_sync_result(entity_type, "success")

        except Exception as e:
            logger.error(f"Error synchronizing {entity_type}: {e}")

            # Record failed sync
            self._record_sync_result(entity_type, "failed", str(e))

    def _get_sync_direction(self, entity_type: str) -> str:
        """
        Determine sync direction for an entity type

        Args:
            entity_type: Type of entity

        Returns:
            Sync direction ("neo4j_to_postgres" or "postgres_to_neo4j")
        """
        # Define source of truth for each entity type
        sources_of_truth = {
            "technique": "neo4j_to_postgres",
            "tactic": "neo4j_to_postgres",
            "mitigation": "neo4j_to_postgres",
            "group": "neo4j_to_postgres",
            "software": "neo4j_to_postgres",
            "analysis_job": "postgres_to_neo4j",
            "analysis_result": "postgres_to_neo4j",
            "user": "postgres_to_neo4j",
        }

        return sources_of_truth.get(entity_type, "neo4j_to_postgres")

    def _sync_from_neo4j_to_postgres(self, entity_type: str):
        """
        Synchronize data from Neo4j to PostgreSQL

        Args:
            entity_type: Type of entity to synchronize
        """
        # Get last sync time
        last_sync = self._get_last_sync_time(entity_type)

        # Fetch entities from Neo4j
        if entity_type == "technique":
            neo4j_entities = self._fetch_techniques_from_neo4j(last_sync)
        elif entity_type == "tactic":
            neo4j_entities = self._fetch_tactics_from_neo4j(last_sync)
        # Add cases for other entity types...
        else:
            logger.warning(f"Sync not implemented for entity type: {entity_type}")
            return

        # Fetch current entities from PostgreSQL for comparison
        postgres_entities = self._fetch_entities_from_postgres(entity_type)

        # Find entities to insert, update, or delete
        to_insert, to_update, to_delete = self._diff_entities(
            neo4j_entities, postgres_entities
        )

        # Perform operations
        inserted = self._insert_entities_to_postgres(entity_type, to_insert)
        updated = self._update_entities_in_postgres(entity_type, to_update)
        deleted = self._delete_entities_from_postgres(entity_type, to_delete)

        logger.info(
            f"Synced {entity_type} from Neo4j to PostgreSQL: "
            f"{inserted} inserted, {updated} updated, {deleted} deleted"
        )

    def _sync_from_postgres_to_neo4j(self, entity_type: str):
        """
        Synchronize data from PostgreSQL to Neo4j

        Args:
            entity_type: Type of entity to synchronize
        """
        # Get last sync time
        last_sync = self._get_last_sync_time(entity_type)

        # Fetch entities from PostgreSQL
        if entity_type == "analysis_job":
            postgres_entities = self._fetch_analysis_jobs_from_postgres(last_sync)
        elif entity_type == "analysis_result":
            postgres_entities = self._fetch_analysis_results_from_postgres(last_sync)
        # Add cases for other entity types...
        else:
            logger.warning(f"Sync not implemented for entity type: {entity_type}")
            return

        # Fetch current entities from Neo4j for comparison
        neo4j_entities = self._fetch_entities_from_neo4j(entity_type)

        # Find entities to insert, update, or delete
        to_insert, to_update, to_delete = self._diff_entities(
            postgres_entities, neo4j_entities
        )

        # Perform operations
        inserted = self._insert_entities_to_neo4j(entity_type, to_insert)
        updated = self._update_entities_in_neo4j(entity_type, to_update)
        deleted = self._delete_entities_from_neo4j(entity_type, to_delete)

        logger.info(
            f"Synced {entity_type} from PostgreSQL to Neo4j: "
            f"{inserted} inserted, {updated} updated, {deleted} deleted"
        )

    def _get_last_sync_time(self, entity_type: str) -> Optional[datetime]:
        """
        Get the last sync time for an entity type

        Args:
            entity_type: Type of entity

        Returns:
            Last sync time or None if never synced
        """
        result = self.postgres.query_one(
            """
            SELECT last_sync_time 
            FROM sync_metadata 
            WHERE entity_type = %s
            ORDER BY last_sync_time DESC
            LIMIT 1
            """,
            (entity_type,),
        )

        if result and "last_sync_time" in result:
            return result["last_sync_time"]

        return None

    def _record_sync_result(
        self, entity_type: str, status: str, error_message: str = None
    ):
        """
        Record sync result in the metadata table

        Args:
            entity_type: Type of entity synced
            status: Status of the sync operation
            error_message: Error message if sync failed
        """
        self.postgres.execute(
            """
            INSERT INTO sync_metadata
            (entity_type, last_sync_time, entity_count, status, error_message)
            VALUES (%s, CURRENT_TIMESTAMP, %s, %s, %s)
            """,
            (entity_type, 0, status, error_message),  # Will update with actual count
        )

    # Continuing src/database/sync_manager.py

    def _fetch_techniques_from_neo4j(self, last_sync: Optional[datetime]) -> List[Dict]:
        """
        Fetch technique entities from Neo4j

        Args:
            last_sync: Timestamp of last sync

        Returns:
            List of technique entities
        """
        query = """
        MATCH (t:AttackTechnique)
        OPTIONAL MATCH (t)-[:BELONGS_TO]->(tactic:AttackTactic)
        WHERE t.updated_at IS NULL OR t.updated_at > datetime($last_sync)
        RETURN 
            t.technique_id as id,
            t.name as name,
            t.description as description,
            t.is_subtechnique as is_subtechnique,
            t.parent_technique_id as parent_technique_id,
            t.url as url,
            collect(DISTINCT tactic.name) as tactics,
            t.updated_at as updated_at,
            apoc.convert.toJson(t{.*}) as raw_data
        """

        params = {
            "last_sync": last_sync.isoformat() if last_sync else "1970-01-01T00:00:00"
        }

        results = self.neo4j.run_query(query, params)

        # Add entity_hash for comparison
        for result in results:
            result["entity_hash"] = self._calculate_entity_hash(result)

        return results

    def _fetch_tactics_from_neo4j(self, last_sync: Optional[datetime]) -> List[Dict]:
        """
        Fetch tactic entities from Neo4j

        Args:
            last_sync: Timestamp of last sync

        Returns:
            List of tactic entities
        """
        query = """
        MATCH (t:AttackTactic)
        WHERE t.updated_at IS NULL OR t.updated_at > datetime($last_sync)
        RETURN 
            t.tactic_id as id,
            t.name as name,
            t.description as description,
            t.url as url,
            t.updated_at as updated_at,
            apoc.convert.toJson(t{.*}) as raw_data
        """

        params = {
            "last_sync": last_sync.isoformat() if last_sync else "1970-01-01T00:00:00"
        }

        results = self.neo4j.run_query(query, params)

        # Add entity_hash for comparison
        for result in results:
            result["entity_hash"] = self._calculate_entity_hash(result)

        return results

    def _fetch_analysis_jobs_from_postgres(
        self, last_sync: Optional[datetime]
    ) -> List[Dict]:
        """
        Fetch analysis job entities from PostgreSQL

        Args:
            last_sync: Timestamp of last sync

        Returns:
            List of analysis job entities
        """
        if last_sync:
            query = """
            SELECT id, user_id, name, status, input_type, extractors_used, 
                   threshold, created_at, completed_at, processing_time_ms
            FROM analysis_jobs
            WHERE created_at > %s OR 
                  (status = 'completed' AND completed_at > %s) OR
                  (status = 'failed' AND updated_at > %s)
            """
            params = (last_sync, last_sync, last_sync)
        else:
            query = """
            SELECT id, user_id, name, status, input_type, extractors_used, 
                   threshold, created_at, completed_at, processing_time_ms
            FROM analysis_jobs
            """
            params = None

        results = self.postgres.query(query, params)

        # Add entity_hash for comparison
        for result in results:
            # Convert to dict if not already
            if not isinstance(result, dict):
                result = dict(result)

            result["entity_hash"] = self._calculate_entity_hash(result)

        return results

    def _fetch_analysis_results_from_postgres(
        self, last_sync: Optional[datetime]
    ) -> List[Dict]:
        """
        Fetch analysis result entities from PostgreSQL

        Args:
            last_sync: Timestamp of last sync

        Returns:
            List of analysis result entities
        """
        if last_sync:
            query = """
            SELECT id, job_id, technique_id, technique_name, confidence, 
                   method, matched_keywords, cve_id, created_at
            FROM analysis_results
            WHERE created_at > %s
            """
            params = (last_sync,)
        else:
            query = """
            SELECT id, job_id, technique_id, technique_name, confidence, 
                   method, matched_keywords, cve_id, created_at
            FROM analysis_results
            """
            params = None

        results = self.postgres.query(query, params)

        # Add entity_hash for comparison
        for result in results:
            # Convert to dict if not already
            if not isinstance(result, dict):
                result = dict(result)

            result["entity_hash"] = self._calculate_entity_hash(result)

        return results

    def _fetch_entities_from_postgres(self, entity_type: str) -> Dict[str, Dict]:
        """
        Fetch entities from PostgreSQL for comparison

        Args:
            entity_type: Type of entity

        Returns:
            Dictionary mapping entity IDs to entity data
        """
        if entity_type == "technique":
            query = """
            SELECT id, name, description, is_subtechnique, parent_technique_id, 
                   url, tactics, entity_hash, created_at, updated_at
            FROM attack_techniques
            """
        elif entity_type == "tactic":
            query = """
            SELECT id, name, description, url, entity_hash, created_at, updated_at
            FROM attack_tactics
            """
        else:
            logger.warning(f"Fetch not implemented for entity type: {entity_type}")
            return {}

        results = self.postgres.query(query)

        # Create dictionary of ID -> entity
        entities = {}
        for result in results:
            # Convert to dict if not already
            if not isinstance(result, dict):
                result = dict(result)

            entities[result["id"]] = result

        return entities

    def _fetch_entities_from_neo4j(self, entity_type: str) -> Dict[str, Dict]:
        """
        Fetch entities from Neo4j for comparison

        Args:
            entity_type: Type of entity

        Returns:
            Dictionary mapping entity IDs to entity data
        """
        if entity_type == "analysis_job":
            query = """
            MATCH (j:AnalysisJob)
            RETURN 
                j.job_id as id,
                j.user_id as user_id,
                j.status as status,
                j.input_type as input_type,
                j.extractors_used as extractors_used,
                j.threshold as threshold,
                j.created_at as created_at,
                j.completed_at as completed_at,
                j.processing_time_ms as processing_time_ms,
                apoc.convert.toJson(j{.*}) as raw_data
            """
        elif entity_type == "analysis_result":
            query = """
            MATCH (r:AnalysisResult)
            RETURN 
                r.result_id as id,
                r.job_id as job_id,
                r.technique_id as technique_id,
                r.confidence as confidence,
                r.method as method,
                r.matched_keywords as matched_keywords,
                r.cve_id as cve_id,
                r.created_at as created_at,
                apoc.convert.toJson(r{.*}) as raw_data
            """
        else:
            logger.warning(f"Fetch not implemented for entity type: {entity_type}")
            return {}

        results = self.neo4j.run_query(query)

        # Create dictionary of ID -> entity
        entities = {}
        for result in results:
            # Calculate entity hash
            result["entity_hash"] = self._calculate_entity_hash(result)
            entities[result["id"]] = result

        return entities

    def _diff_entities(
        self, source_entities: List[Dict], target_entities: Dict[str, Dict]
    ) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        Find entities to insert, update, or delete

        Args:
            source_entities: List of entities from source database
            target_entities: Dictionary of entities from target database

        Returns:
            Tuple of (entities to insert, entities to update, entity IDs to delete)
        """
        to_insert = []
        to_update = []
        existing_ids = set()

        for entity in source_entities:
            entity_id = entity["id"]
            existing_ids.add(entity_id)

            if entity_id not in target_entities:
                # Entity doesn't exist in target, insert it
                to_insert.append(entity)
            elif entity.get("entity_hash") != target_entities[entity_id].get(
                "entity_hash"
            ):
                # Entity exists but has changed, update it
                to_update.append(entity)

        # Find entities in target that don't exist in source
        to_delete = [
            entity_id
            for entity_id in target_entities.keys()
            if entity_id not in existing_ids
        ]

        return to_insert, to_update, to_delete

    def _calculate_entity_hash(self, entity: Dict) -> str:
        """
        Calculate hash of entity data for comparison

        Args:
            entity: Entity dictionary

        Returns:
            Hash string
        """
        # Create a copy with only the fields we want to hash
        hash_data = {}

        for key, value in entity.items():
            # Skip metadata and hash fields
            if key in ["created_at", "updated_at", "entity_hash", "raw_data"]:
                continue

            hash_data[key] = value

        # Convert to stable JSON representation
        json_str = json.dumps(hash_data, sort_keys=True)

        # Calculate hash
        return hashlib.md5(json_str.encode()).hexdigest()

    def _insert_entities_to_postgres(
        self, entity_type: str, entities: List[Dict]
    ) -> int:
        """
        Insert entities into PostgreSQL

        Args:
            entity_type: Type of entity
            entities: List of entities to insert

        Returns:
            Number of entities inserted
        """
        if not entities:
            return 0

        inserted = 0

        if entity_type == "technique":
            for entity in entities:
                self.postgres.execute(
                    """
                    INSERT INTO attack_techniques 
                    (id, name, description, is_subtechnique, parent_technique_id, url, tactics, entity_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entity["id"],
                        entity["name"],
                        entity["description"],
                        entity["is_subtechnique"],
                        entity["parent_technique_id"],
                        entity["url"],
                        entity.get("tactics", []),
                        entity["entity_hash"],
                    ),
                )
                inserted += 1
        elif entity_type == "tactic":
            for entity in entities:
                self.postgres.execute(
                    """
                    INSERT INTO attack_tactics 
                    (id, name, description, url, entity_hash)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        entity["id"],
                        entity["name"],
                        entity["description"],
                        entity["url"],
                        entity["entity_hash"],
                    ),
                )
                inserted += 1
        else:
            logger.warning(f"Insert not implemented for entity type: {entity_type}")

        return inserted

    def _update_entities_in_postgres(
        self, entity_type: str, entities: List[Dict]
    ) -> int:
        """
        Update entities in PostgreSQL

        Args:
            entity_type: Type of entity
            entities: List of entities to update

        Returns:
            Number of entities updated
        """
        if not entities:
            return 0

        updated = 0

        if entity_type == "technique":
            for entity in entities:
                self.postgres.execute(
                    """
                    UPDATE attack_techniques 
                    SET name = %s, 
                        description = %s, 
                        is_subtechnique = %s, 
                        parent_technique_id = %s, 
                        url = %s, 
                        tactics = %s, 
                        entity_hash = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (
                        entity["name"],
                        entity["description"],
                        entity["is_subtechnique"],
                        entity["parent_technique_id"],
                        entity["url"],
                        entity.get("tactics", []),
                        entity["entity_hash"],
                        entity["id"],
                    ),
                )
                updated += 1
        elif entity_type == "tactic":
            for entity in entities:
                self.postgres.execute(
                    """
                    UPDATE attack_tactics 
                    SET name = %s, 
                        description = %s, 
                        url = %s, 
                        entity_hash = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (
                        entity["name"],
                        entity["description"],
                        entity["url"],
                        entity["entity_hash"],
                        entity["id"],
                    ),
                )
                updated += 1
        else:
            logger.warning(f"Update not implemented for entity type: {entity_type}")

        return updated

    def _delete_entities_from_postgres(
        self, entity_type: str, entity_ids: List[str]
    ) -> int:
        """
        Delete entities from PostgreSQL

        Args:
            entity_type: Type of entity
            entity_ids: List of entity IDs to delete

        Returns:
            Number of entities deleted
        """
        if not entity_ids:
            return 0

        table_name = None

        if entity_type == "technique":
            table_name = "attack_techniques"
        elif entity_type == "tactic":
            table_name = "attack_tactics"
        else:
            logger.warning(f"Delete not implemented for entity type: {entity_type}")
            return 0

        # Delete entities
        format_strings = ",".join(["%s"] * len(entity_ids))
        query = f"DELETE FROM {table_name} WHERE id IN ({format_strings})"

        result = self.postgres.execute(query, entity_ids)

        return result

    def _insert_entities_to_neo4j(self, entity_type: str, entities: List[Dict]) -> int:
        """
        Insert entities into Neo4j

        Args:
            entity_type: Type of entity
            entities: List of entities to insert

        Returns:
            Number of entities inserted
        """
        if not entities:
            return 0

        inserted = 0

        if entity_type == "analysis_job":
            for entity in entities:
                query = """
                CREATE (j:AnalysisJob {
                    job_id: $id,
                    user_id: $user_id,
                    status: $status,
                    input_type: $input_type,
                    extractors_used: $extractors_used,
                    threshold: $threshold,
                    created_at: datetime($created_at),
                    completed_at: CASE WHEN $completed_at IS NULL THEN null ELSE datetime($completed_at) END,
                    processing_time_ms: $processing_time_ms,
                    entity_hash: $entity_hash
                })
                """

                params = {
                    "id": entity["id"],
                    "user_id": entity["user_id"],
                    "status": entity["status"],
                    "input_type": entity["input_type"],
                    "extractors_used": entity["extractors_used"],
                    "threshold": entity["threshold"],
                    "created_at": entity["created_at"].isoformat()
                    if entity["created_at"]
                    else None,
                    "completed_at": entity["completed_at"].isoformat()
                    if entity["completed_at"]
                    else None,
                    "processing_time_ms": entity["processing_time_ms"],
                    "entity_hash": entity["entity_hash"],
                }

                self.neo4j.run_query(query, params)
                inserted += 1
        elif entity_type == "analysis_result":
            for entity in entities:
                query = """
                MATCH (j:AnalysisJob {job_id: $job_id})
                MATCH (t:AttackTechnique {technique_id: $technique_id})
                CREATE (j)-[:PRODUCED]->(r:AnalysisResult {
                    result_id: $id,
                    confidence: $confidence,
                    method: $method,
                    matched_keywords: $matched_keywords,
                    cve_id: $cve_id,
                    created_at: datetime($created_at),
                    entity_hash: $entity_hash
                })-[:IDENTIFIES]->(t)
                """

                params = {
                    "id": entity["id"],
                    "job_id": entity["job_id"],
                    "technique_id": entity["technique_id"],
                    "confidence": entity["confidence"],
                    "method": entity["method"],
                    "matched_keywords": entity["matched_keywords"],
                    "cve_id": entity["cve_id"],
                    "created_at": entity["created_at"].isoformat()
                    if entity["created_at"]
                    else None,
                    "entity_hash": entity["entity_hash"],
                }

                try:
                    self.neo4j.run_query(query, params)
                    inserted += 1
                except Exception as e:
                    # Log error but continue with other entities
                    logger.error(f"Error inserting analysis result: {e}")
        else:
            logger.warning(f"Insert not implemented for entity type: {entity_type}")

        return inserted

    def _update_entities_in_neo4j(self, entity_type: str, entities: List[Dict]) -> int:
        """
        Update entities in Neo4j

        Args:
            entity_type: Type of entity
            entities: List of entities to update

        Returns:
            Number of entities updated
        """
        if not entities:
            return 0

        updated = 0

        if entity_type == "analysis_job":
            for entity in entities:
                query = """
                MATCH (j:AnalysisJob {job_id: $id})
                SET j.user_id = $user_id,
                    j.status = $status,
                    j.input_type = $input_type,
                    j.extractors_used = $extractors_used,
                    j.threshold = $threshold,
                    j.created_at = datetime($created_at),
                    j.completed_at = CASE WHEN $completed_at IS NULL THEN null ELSE datetime($completed_at) END,
                    j.processing_time_ms = $processing_time_ms,
                    j.entity_hash = $entity_hash,
                    j.updated_at = datetime()
                """

                params = {
                    "id": entity["id"],
                    "user_id": entity["user_id"],
                    "status": entity["status"],
                    "input_type": entity["input_type"],
                    "extractors_used": entity["extractors_used"],
                    "threshold": entity["threshold"],
                    "created_at": entity["created_at"].isoformat()
                    if entity["created_at"]
                    else None,
                    "completed_at": entity["completed_at"].isoformat()
                    if entity["completed_at"]
                    else None,
                    "processing_time_ms": entity["processing_time_ms"],
                    "entity_hash": entity["entity_hash"],
                }

                self.neo4j.run_query(query, params)
                updated += 1
        elif entity_type == "analysis_result":
            # For analysis results, we'll delete and recreate since relationships might change
            for entity in entities:
                # Delete existing result
                delete_query = """
                MATCH (r:AnalysisResult {result_id: $id})
                DETACH DELETE r
                """

                self.neo4j.run_query(delete_query, {"id": entity["id"]})

                # Create new result with updated data
                create_query = """
                MATCH (j:AnalysisJob {job_id: $job_id})
                MATCH (t:AttackTechnique {technique_id: $technique_id})
                CREATE (j)-[:PRODUCED]->(r:AnalysisResult {
                    result_id: $id,
                    confidence: $confidence,
                    method: $method,
                    matched_keywords: $matched_keywords,
                    cve_id: $cve_id,
                    created_at: datetime($created_at),
                    entity_hash: $entity_hash,
                    updated_at: datetime()
                })-[:IDENTIFIES]->(t)
                """

                params = {
                    "id": entity["id"],
                    "job_id": entity["job_id"],
                    "technique_id": entity["technique_id"],
                    "confidence": entity["confidence"],
                    "method": entity["method"],
                    "matched_keywords": entity["matched_keywords"],
                    "cve_id": entity["cve_id"],
                    "created_at": entity["created_at"].isoformat()
                    if entity["created_at"]
                    else None,
                    "entity_hash": entity["entity_hash"],
                }

                try:
                    self.neo4j.run_query(create_query, params)
                    updated += 1
                except Exception as e:
                    # Log error but continue with other entities
                    logger.error(f"Error updating analysis result: {e}")
        else:
            logger.warning(f"Update not implemented for entity type: {entity_type}")

        return updated

    def _delete_entities_from_neo4j(
        self, entity_type: str, entity_ids: List[str]
    ) -> int:
        """
        Delete entities from Neo4j

        Args:
            entity_type: Type of entity
            entity_ids: List of entity IDs to delete

        Returns:
            Number of entities deleted
        """
        if not entity_ids:
            return 0

        delete_query = None

        if entity_type == "analysis_job":
            delete_query = """
            UNWIND $ids AS id
            MATCH (j:AnalysisJob {job_id: id})
            DETACH DELETE j
            """
        elif entity_type == "analysis_result":
            delete_query = """
            UNWIND $ids AS id
            MATCH (r:AnalysisResult {result_id: id})
            DETACH DELETE r
            """
        else:
            logger.warning(f"Delete not implemented for entity type: {entity_type}")
            return 0

        # Delete entities
        self.neo4j.run_query(delete_query, {"ids": entity_ids})

        return len(entity_ids)
