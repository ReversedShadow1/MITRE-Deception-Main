import csv
import hashlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import schedule
from neo4j import GraphDatabase

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"mitre_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logger = logging.getLogger("mitre_integration")


class Neo4jConnector:
    """Handles connections and operations with the Neo4j database."""

    def __init__(self, uri: str, user: str, password: str):
        """Initialize the Neo4j connector.

        Args:
            uri: The URI for the Neo4j instance
            user: Username for the Neo4j instance
            password: Password for the Neo4j instance
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self) -> bool:
        """Connect to the Neo4j database.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password), database="pipe"
            )
            logger.info("Successfully connected to Neo4j database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            return False

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def run_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Run a Cypher query against the Neo4j database.

        Args:
            query: The Cypher query to run
            params: Parameters for the query

        Returns:
            List of dictionaries with query results
        """
        if not self.driver:
            logger.error("No active connection to Neo4j")
            return []

        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return []

    def create_constraints_and_indexes(self) -> None:
        """Create necessary constraints and indexes for performance."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:AttackTechnique) REQUIRE t.technique_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:AttackTactic) REQUIRE t.tactic_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:AttackGroup) REQUIRE g.group_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:AttackSoftware) REQUIRE s.software_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:AttackMitigation) REQUIRE m.mitigation_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:D3FENDTechnique) REQUIRE d.d3fend_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:EngageTechnique) REQUIRE e.engage_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:CWE) REQUIRE c.cwe_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:CAPEC) REQUIRE p.capec_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:CVE) REQUIRE v.cve_id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (t:AttackTechnique) ON (t.name)",
            "CREATE INDEX IF NOT EXISTS FOR (v:CVE) ON (v.published_date)",
        ]

        for constraint in constraints:
            try:
                self.run_query(constraint)
                logger.info(f"Successfully created: {constraint}")
            except Exception as e:
                logger.error(f"Error creating constraint/index: {e}")


class MitreAttackConnector:
    """Handles retrieval and processing of MITRE ATT&CK data."""

    def __init__(self, neo4j_connector: Neo4jConnector):
        """Initialize the MITRE ATT&CK connector.

        Args:
            neo4j_connector: An initialized Neo4j connector
        """
        self.neo4j = neo4j_connector
        self.attack_base_url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"

    def fetch_attack_data(self) -> Dict:
        """Fetch MITRE ATT&CK data from the MITRE CTI GitHub repository.

        Returns:
            Dict containing ATT&CK data
        """
        logger.info("Fetching MITRE ATT&CK data...")
        try:
            response = requests.get(self.attack_base_url, timeout=60)
            response.raise_for_status()
            logger.info("Successfully fetched MITRE ATT&CK data")
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching MITRE ATT&CK data: {e}")
            return {}

    def process_attack_data(self, attack_data: Dict) -> None:
        """Process and insert MITRE ATT&CK data into Neo4j.

        Args:
            attack_data: ATT&CK data dictionary
        """
        if not attack_data or "objects" not in attack_data:
            logger.error("Invalid ATT&CK data format")
            return

        logger.info("Processing MITRE ATT&CK data...")

        # Extract different types of objects
        objects = attack_data.get("objects", [])

        # Process tactics (kill chain phases)
        tactics = {}
        for obj in objects:
            if obj.get("type") == "x-mitre-tactic":
                tactic_id = obj.get("external_references", [{}])[0].get(
                    "external_id", ""
                )
                if tactic_id:
                    tactics[obj.get("id")] = {
                        "tactic_id": tactic_id,
                        "name": obj.get("name", ""),
                        "description": obj.get("description", ""),
                    }

        # Insert tactics
        for tactic_stix_id, tactic_data in tactics.items():
            query = """
            MERGE (t:AttackTactic {tactic_id: $tactic_id})
            ON CREATE SET 
                t.name = $name,
                t.description = $description,
                t.stix_id = $stix_id,
                t.created_at = datetime(),
                t.updated_at = datetime()
            ON MATCH SET 
                t.name = $name,
                t.description = $description,
                t.stix_id = $stix_id,
                t.updated_at = datetime()
            """
            params = {
                "tactic_id": tactic_data["tactic_id"],
                "name": tactic_data["name"],
                "description": tactic_data["description"],
                "stix_id": tactic_stix_id,
            }
            self.neo4j.run_query(query, params)

        logger.info(f"Processed {len(tactics)} ATT&CK tactics")

        # Process techniques and sub-techniques
        techniques_count = 0
        for obj in objects:
            if obj.get("type") == "attack-pattern":
                external_refs = obj.get("external_references", [])
                technique_id = ""
                url = ""
                for ref in external_refs:
                    if ref.get("source_name") == "mitre-attack":
                        technique_id = ref.get("external_id", "")
                        url = ref.get("url", "")
                        break

                if not technique_id:
                    continue

                # Get kill chain phases (tactics)
                kill_chain_phases = obj.get("kill_chain_phases", [])
                tactic_ids = []
                for phase in kill_chain_phases:
                    if phase.get("kill_chain_name") == "mitre-attack":
                        tactic_ids.append(phase.get("phase_name", ""))

                # Determine if it's a sub-technique
                is_subtechnique = "." in technique_id
                parent_technique_id = (
                    technique_id.split(".")[0] if is_subtechnique else ""
                )

                # Create the technique node
                query = """
                MERGE (t:AttackTechnique {technique_id: $technique_id})
                ON CREATE SET 
                    t.name = $name,
                    t.description = $description,
                    t.stix_id = $stix_id,
                    t.is_subtechnique = $is_subtechnique,
                    t.parent_technique_id = $parent_technique_id,
                    t.url = $url,
                    t.created_at = datetime(),
                    t.updated_at = datetime(),
                    t.detection = $detection,
                    t.data_sources = $data_sources,
                    t.platforms = $platforms
                ON MATCH SET 
                    t.name = $name,
                    t.description = $description,
                    t.stix_id = $stix_id,
                    t.is_subtechnique = $is_subtechnique,
                    t.parent_technique_id = $parent_technique_id,
                    t.url = $url,
                    t.updated_at = datetime(),
                    t.detection = $detection,
                    t.data_sources = $data_sources,
                    t.platforms = $platforms
                """
                params = {
                    "technique_id": technique_id,
                    "name": obj.get("name", ""),
                    "description": obj.get("description", ""),
                    "stix_id": obj.get("id", ""),
                    "is_subtechnique": is_subtechnique,
                    "parent_technique_id": parent_technique_id,
                    "url": url,
                    "detection": obj.get("x_mitre_detection", ""),
                    "data_sources": json.dumps(obj.get("x_mitre_data_sources", [])),
                    "platforms": json.dumps(obj.get("x_mitre_platforms", [])),
                }
                self.neo4j.run_query(query, params)

                # Create relationships to tactics
                for tactic_name in tactic_ids:
                    query = """
                    MATCH (t:AttackTechnique {technique_id: $technique_id})
                    MATCH (a:AttackTactic) WHERE a.name = $tactic_name
                    MERGE (t)-[r:BELONGS_TO]->(a)
                    ON CREATE SET r.created_at = datetime()
                    """
                    params = {"technique_id": technique_id, "tactic_name": tactic_name}
                    self.neo4j.run_query(query, params)

                # If it's a sub-technique, create a relationship to the parent
                if is_subtechnique:
                    query = """
                    MATCH (sub:AttackTechnique {technique_id: $subtechnique_id})
                    MATCH (parent:AttackTechnique {technique_id: $parent_id})
                    MERGE (sub)-[r:SUBTECHNIQUE_OF]->(parent)
                    ON CREATE SET r.created_at = datetime()
                    """
                    params = {
                        "subtechnique_id": technique_id,
                        "parent_id": parent_technique_id,
                    }
                    self.neo4j.run_query(query, params)

                techniques_count += 1

        logger.info(
            f"Processed {techniques_count} ATT&CK techniques and sub-techniques"
        )

        # Process groups (intrusion sets)
        groups_count = 0
        for obj in objects:
            if obj.get("type") == "intrusion-set":
                external_refs = obj.get("external_references", [])
                group_id = ""
                url = ""
                for ref in external_refs:
                    if ref.get("source_name") == "mitre-attack":
                        group_id = ref.get("external_id", "")
                        url = ref.get("url", "")
                        break

                if not group_id:
                    continue

                query = """
                MERGE (g:AttackGroup {group_id: $group_id})
                ON CREATE SET 
                    g.name = $name,
                    g.description = $description,
                    g.stix_id = $stix_id,
                    g.url = $url,
                    g.aliases = $aliases,
                    g.created_at = datetime(),
                    g.updated_at = datetime()
                ON MATCH SET 
                    g.name = $name,
                    g.description = $description,
                    g.stix_id = $stix_id,
                    g.url = $url,
                    g.aliases = $aliases,
                    g.updated_at = datetime()
                """
                params = {
                    "group_id": group_id,
                    "name": obj.get("name", ""),
                    "description": obj.get("description", ""),
                    "stix_id": obj.get("id", ""),
                    "url": url,
                    "aliases": json.dumps(obj.get("aliases", [])),
                }
                self.neo4j.run_query(query, params)
                groups_count += 1

        logger.info(f"Processed {groups_count} ATT&CK groups")

        # Process software (tools and malware)
        software_count = 0
        for obj in objects:
            if obj.get("type") in ["tool", "malware"]:
                external_refs = obj.get("external_references", [])
                software_id = ""
                url = ""
                for ref in external_refs:
                    if ref.get("source_name") == "mitre-attack":
                        software_id = ref.get("external_id", "")
                        url = ref.get("url", "")
                        break

                if not software_id:
                    continue

                query = """
                MERGE (s:AttackSoftware {software_id: $software_id})
                ON CREATE SET 
                    s.name = $name,
                    s.description = $description,
                    s.stix_id = $stix_id,
                    s.type = $type,
                    s.url = $url,
                    s.platforms = $platforms,
                    s.created_at = datetime(),
                    s.updated_at = datetime()
                ON MATCH SET 
                    s.name = $name,
                    s.description = $description,
                    s.stix_id = $stix_id,
                    s.type = $type,
                    s.url = $url,
                    s.platforms = $platforms,
                    s.updated_at = datetime()
                """
                params = {
                    "software_id": software_id,
                    "name": obj.get("name", ""),
                    "description": obj.get("description", ""),
                    "stix_id": obj.get("id", ""),
                    "type": obj.get("type", ""),
                    "url": url,
                    "platforms": json.dumps(obj.get("x_mitre_platforms", [])),
                }
                self.neo4j.run_query(query, params)
                software_count += 1

        logger.info(f"Processed {software_count} ATT&CK software items")

        # Process relationships
        relationships_count = 0
        for obj in objects:
            if obj.get("type") == "relationship":
                source_ref = obj.get("source_ref", "")
                target_ref = obj.get("target_ref", "")
                relationship_type = (
                    obj.get("relationship_type", "").upper().replace("-", "_")
                )

                if not (source_ref and target_ref and relationship_type):
                    continue

                query = """
                MATCH (source) WHERE source.stix_id = $source_ref
                MATCH (target) WHERE target.stix_id = $target_ref
                CALL apoc.merge.relationship(source, $relationship_type, 
                    {stix_id: $stix_id}, 
                    {description: $description, created_at: datetime()}, 
                    target) 
                YIELD rel
                RETURN rel
                """
                params = {
                    "source_ref": source_ref,
                    "target_ref": target_ref,
                    "relationship_type": relationship_type,
                    "stix_id": obj.get("id", ""),
                    "description": obj.get("description", ""),
                }
                try:
                    self.neo4j.run_query(query, params)
                    relationships_count += 1
                except Exception as e:
                    # If APOC is not available, use a simpler approach
                    logger.warning(f"APOC error, using fallback: {e}")
                    rel_query = f"""
                    MATCH (source) WHERE source.stix_id = $source_ref
                    MATCH (target) WHERE target.stix_id = $target_ref
                    MERGE (source)-[r:{relationship_type}]->(target)
                    ON CREATE SET 
                        r.stix_id = $stix_id,
                        r.description = $description,
                        r.created_at = datetime()
                    """
                    self.neo4j.run_query(rel_query, params)
                    relationships_count += 1

        logger.info(f"Processed {relationships_count} ATT&CK relationships")

        # Process mitigations
        mitigations_count = 0
        for obj in objects:
            if obj.get("type") == "course-of-action":
                external_refs = obj.get("external_references", [])
                mitigation_id = ""
                url = ""
                for ref in external_refs:
                    if ref.get("source_name") == "mitre-attack":
                        mitigation_id = ref.get("external_id", "")
                        url = ref.get("url", "")
                        break

                if not mitigation_id:
                    continue

                query = """
                MERGE (m:AttackMitigation {mitigation_id: $mitigation_id})
                ON CREATE SET 
                    m.name = $name,
                    m.description = $description,
                    m.stix_id = $stix_id,
                    m.url = $url,
                    m.created_at = datetime(),
                    m.updated_at = datetime()
                ON MATCH SET 
                    m.name = $name,
                    m.description = $description,
                    m.stix_id = $stix_id,
                    m.url = $url,
                    m.updated_at = datetime()
                """
                params = {
                    "mitigation_id": mitigation_id,
                    "name": obj.get("name", ""),
                    "description": obj.get("description", ""),
                    "stix_id": obj.get("id", ""),
                    "url": url,
                }
                self.neo4j.run_query(query, params)
                mitigations_count += 1

        logger.info(f"Processed {mitigations_count} ATT&CK mitigations")
        logger.info("Completed processing MITRE ATT&CK data")

    def update(self) -> None:
        """Update ATT&CK data in the database."""
        attack_data = self.fetch_attack_data()
        if attack_data:
            self.process_attack_data(attack_data)


class D3FENDConnector:
    """Handles processing of D3FEND data from CSV."""

    def __init__(self, neo4j_connector: Neo4jConnector):
        """Initialize the D3FEND connector.

        Args:
            neo4j_connector: An initialized Neo4j connector
        """
        self.neo4j = neo4j_connector
        self.d3fend_csv_path = "data/d3fend-full-mappings.csv"

    def process_d3fend_data(self) -> None:
        """Process and insert D3FEND data from CSV into Neo4j."""
        logger.info(f"Processing D3FEND data from {self.d3fend_csv_path}...")

        try:
            d3fend_df = pd.read_csv(self.d3fend_csv_path)
            logger.info(f"Loaded {len(d3fend_df)} rows from D3FEND CSV file")

            # Process D3FEND techniques
            techniques_processed = set()
            for _, row in d3fend_df.iterrows():
                # Extract technique details
                technique_id = self._extract_id_from_uri(row["def_tech"])
                technique_name = row["def_tech_label"]
                top_technique_name = row["top_def_tech_label"]

                if not technique_id or technique_id in techniques_processed:
                    continue

                # Create D3FEND technique node
                query = """
                MERGE (d:D3FENDTechnique {d3fend_id: $d3fend_id})
                ON CREATE SET 
                    d.name = $name,
                    d.top_technique_name = $top_technique_name,
                    d.uri = $uri,
                    d.created_at = datetime(),
                    d.updated_at = datetime()
                ON MATCH SET 
                    d.name = $name,
                    d.top_technique_name = $top_technique_name,
                    d.uri = $uri,
                    d.updated_at = datetime()
                """
                params = {
                    "d3fend_id": technique_id,
                    "name": technique_name,
                    "top_technique_name": top_technique_name,
                    "uri": row["def_tech"],
                }
                self.neo4j.run_query(query, params)
                techniques_processed.add(technique_id)

            logger.info(
                f"Processed {len(techniques_processed)} unique D3FEND techniques"
            )

            # Process relationships to ATT&CK techniques
            relationships_count = 0
            for _, row in d3fend_df.iterrows():
                d3fend_id = self._extract_id_from_uri(row["def_tech"])
                attack_id = row["off_tech_id"]

                if not d3fend_id or not attack_id:
                    continue

                # Create relationship from D3FEND technique to ATT&CK technique
                query = """
                MATCH (d:D3FENDTechnique {d3fend_id: $d3fend_id})
                MATCH (a:AttackTechnique {technique_id: $attack_id})
                MERGE (d)-[r:COUNTERS]->(a)
                ON CREATE SET r.created_at = datetime()
                """
                params = {"d3fend_id": d3fend_id, "attack_id": attack_id}
                self.neo4j.run_query(query, params)
                relationships_count += 1

            logger.info(
                f"Processed {relationships_count} D3FEND-to-ATT&CK relationships"
            )

            # Process D3FEND defensive artifacts
            artifacts_processed = set()
            artifacts_count = 0
            for _, row in d3fend_df.iterrows():
                if pd.notna(row["def_artifact"]) and pd.notna(
                    row["def_artifact_label"]
                ):
                    artifact_id = self._extract_id_from_uri(row["def_artifact"])
                    artifact_name = row["def_artifact_label"]

                    if not artifact_id or artifact_id in artifacts_processed:
                        continue

                    # Create D3FEND artifact node
                    query = """
                    MERGE (a:D3FENDArtifact {artifact_id: $artifact_id})
                    ON CREATE SET 
                        a.name = $name,
                        a.uri = $uri,
                        a.created_at = datetime(),
                        a.updated_at = datetime()
                    ON MATCH SET 
                        a.name = $name,
                        a.uri = $uri,
                        a.updated_at = datetime()
                    """
                    params = {
                        "artifact_id": artifact_id,
                        "name": artifact_name,
                        "uri": row["def_artifact"],
                    }
                    self.neo4j.run_query(query, params)
                    artifacts_processed.add(artifact_id)
                    artifacts_count += 1

            logger.info(f"Processed {artifacts_count} D3FEND artifacts")

            # Process relationships between D3FEND techniques and artifacts
            artifact_rel_count = 0
            for _, row in d3fend_df.iterrows():
                if (
                    pd.notna(row["def_artifact"])
                    and pd.notna(row["def_tech"])
                    and pd.notna(row["def_artifact_rel"])
                ):
                    tech_id = self._extract_id_from_uri(row["def_tech"])
                    artifact_id = self._extract_id_from_uri(row["def_artifact"])

                    # Extract the relationship type and replace hyphens with underscores
                    rel_type = self._extract_id_from_uri(row["def_artifact_rel"])
                    if rel_type:
                        rel_type = rel_type.replace("-", "_").upper()

                    if not tech_id or not artifact_id or not rel_type:
                        continue

                    # Create relationship from D3FEND technique to artifact
                    query = f"""
                    MATCH (d:D3FENDTechnique {{d3fend_id: $tech_id}})
                    MATCH (a:D3FENDArtifact {{artifact_id: $artifact_id}})
                    MERGE (d)-[r:{rel_type}]->(a)
                    ON CREATE SET r.created_at = datetime()
                    """
                    params = {"tech_id": tech_id, "artifact_id": artifact_id}
                    self.neo4j.run_query(query, params)
                    artifact_rel_count += 1

            logger.info(
                f"Processed {artifact_rel_count} D3FEND technique-to-artifact relationships"
            )

        except Exception as e:
            logger.error(f"Error processing D3FEND data: {e}")

        logger.info("Completed processing D3FEND data")

    def _extract_id_from_uri(self, uri: str) -> str:
        """Extract an ID from a D3FEND URI.

        Args:
            uri: D3FEND URI

        Returns:
            Extracted ID
        """
        if not uri or not isinstance(uri, str):
            return ""

        # Extract the part after the last # or /
        id_match = re.search(r"[/#]([^/#]+)$", uri)
        if id_match:
            return id_match.group(1)
        return ""

    def update(self) -> None:
        """Update D3FEND data in the database."""
        self.process_d3fend_data()


class MitreEngageConnector:
    """Handles processing of MITRE Engage data from CSV."""

    def __init__(self, neo4j_connector: Neo4jConnector):
        """Initialize the MITRE Engage connector.

        Args:
            neo4j_connector: An initialized Neo4j connector
        """
        self.neo4j = neo4j_connector
        self.engage_csv_path = "data/Engage-Data-V1.0.csv"

    def process_engage_data(self) -> None:
        """Process and insert MITRE Engage data from CSV into Neo4j."""
        logger.info(f"Processing MITRE Engage data from {self.engage_csv_path}...")

        try:
            engage_df = pd.read_csv(self.engage_csv_path)
            logger.info(f"Loaded {len(engage_df)} rows from Engage CSV file")

            # Process Engage activities (EAV values)
            eav_processed = set()
            for _, row in engage_df.iterrows():
                eav_id = row["eav_id"]
                eav_desc = row["eav"]
                eac = row["eac"]
                eac_id = row["eac_id"]

                if not eav_id or eav_id in eav_processed:
                    continue

                # Create Engage technique node
                query = """
                MERGE (e:EngageTechnique {engage_id: $engage_id})
                ON CREATE SET 
                    e.description = $description,
                    e.category = $category,
                    e.category_id = $category_id,
                    e.created = $created,
                    e.modified = $modified,
                    e.version = $version,
                    e.created_at = datetime(),
                    e.updated_at = datetime()
                ON MATCH SET 
                    e.description = $description,
                    e.category = $category,
                    e.category_id = $category_id,
                    e.created = $created,
                    e.modified = $modified,
                    e.version = $version,
                    e.updated_at = datetime()
                """
                params = {
                    "engage_id": eav_id,
                    "description": eav_desc,
                    "category": eac,
                    "category_id": eac_id,
                    "created": row["created"] if pd.notna(row["created"]) else "",
                    "modified": (
                        row["last modified"] if pd.notna(row["last modified"]) else ""
                    ),
                    "version": int(row["version"]) if pd.notna(row["version"]) else 0,
                }
                self.neo4j.run_query(query, params)
                eav_processed.add(eav_id)

            logger.info(f"Processed {len(eav_processed)} unique Engage techniques")

            # Process relationships to ATT&CK techniques
            relationships_count = 0
            for _, row in engage_df.iterrows():
                engage_id = row["eav_id"]
                attack_id = row["attack_id"]

                if not engage_id or not attack_id:
                    continue

                # Create relationship from Engage technique to ATT&CK technique
                query = """
                MATCH (e:EngageTechnique {engage_id: $engage_id})
                MATCH (a:AttackTechnique {technique_id: $attack_id})
                MERGE (e)-[r:ADDRESSES]->(a)
                ON CREATE SET r.created_at = datetime()
                """
                params = {"engage_id": engage_id, "attack_id": attack_id}
                self.neo4j.run_query(query, params)
                relationships_count += 1

            logger.info(
                f"Processed {relationships_count} Engage-to-ATT&CK relationships"
            )

        except Exception as e:
            logger.error(f"Error processing Engage data: {e}")

        logger.info("Completed processing MITRE Engage data")

    def update(self) -> None:
        """Update Engage data in the database."""
        self.process_engage_data()


class CAPECConnector:
    """Handles processing of CAPEC data from CSV."""

    def __init__(self, neo4j_connector: Neo4jConnector):
        """Initialize the CAPEC connector.

        Args:
            neo4j_connector: An initialized Neo4j connector
        """
        self.neo4j = neo4j_connector
        self.capec_csv_path = "data/CAPEC.csv"

    def _parse_example_instances(self, example_instances_str: str) -> List[str]:
        """Parse example instances string into structured data.

        Args:
            example_instances_str: String containing example instances information

        Returns:
            List of example instances
        """
        if (
            not example_instances_str
            or pd.isna(example_instances_str)
            or example_instances_str == ""
        ):
            return []

        examples = []
        # Split by double colons (::) for separate examples
        if "::" in example_instances_str:
            example_parts = example_instances_str.split("::")
            for part in example_parts:
                part = part.strip()
                if part and not part.isspace():
                    examples.append(part)
        else:
            # If no double colons, treat the whole string as one example
            examples.append(example_instances_str.strip())

        return examples

    def _parse_taxonomy_mappings(self, taxonomy_str: str) -> List[Dict]:
        """Parse taxonomy mappings string into structured data."""
        if not taxonomy_str or pd.isna(taxonomy_str) or taxonomy_str == "":
            return []

        mappings = []
        # Updated pattern to better match the actual format
        taxonomy_sections = taxonomy_str.split("::::")

        for section in taxonomy_sections:
            if not section or section.isspace():
                continue

            section = "::" + section + "::" if not section.startswith("::") else section

            # Extract taxonomy name
            taxonomy_match = re.search(r"TAXONOMY NAME:([^:]+)", section)
            entry_id_match = re.search(r"ENTRY ID:([^:]+)", section)
            entry_name_match = re.search(r"ENTRY NAME:([^:]+)", section)

            if taxonomy_match:
                taxonomy_name = taxonomy_match.group(1).strip()
                entry_id = entry_id_match.group(1).strip() if entry_id_match else ""
                entry_name = (
                    entry_name_match.group(1).strip() if entry_name_match else ""
                )

                mappings.append(
                    {
                        "taxonomy": taxonomy_name,
                        "entry_id": entry_id,
                        "entry_name": entry_name,
                    }
                )

        return mappings

    def _extract_attack_mappings(self, mappings: List[Dict]) -> List[str]:
        """Extract ATT&CK technique IDs from taxonomy mappings."""
        attack_ids = []

        for mapping in mappings:
            # More flexible check for ATTACK taxonomy
            if "ATTACK" in mapping["taxonomy"].upper() and mapping["entry_id"]:
                # Extract the actual technique ID (like T1498.001)
                entry_id = mapping["entry_id"]

                # Format the technique ID correctly
                if entry_id.startswith("T"):
                    technique_id = entry_id
                else:
                    # If the entry ID is just a number or starts with a number, prefix with 'T'
                    technique_id = f"T{entry_id}"

                # Log the mapping for debugging
                logger.debug(
                    f"Found ATT&CK mapping: {mapping['taxonomy']} -> {technique_id}"
                )
                attack_ids.append(technique_id)

        return attack_ids

    def process_capec_data(self) -> None:
        """Process and insert CAPEC data from CSV into Neo4j."""
        logger.info(f"Processing CAPEC data from {self.capec_csv_path}...")

        try:
            # Read the CSV file with a low memory footprint approach
            patterns_count = 0
            pattern_ids = set()
            attack_mappings = []

            with open(self.capec_csv_path, "r", encoding="utf-8") as f:
                csv_reader = csv.DictReader(f)

                for row in csv_reader:
                    capec_id = row.get("'ID", "").strip()
                    if not capec_id or capec_id in pattern_ids:
                        continue

                    pattern_ids.add(capec_id)

                    # Extract pattern details
                    name = row.get("Name", "").strip()
                    description = row.get("Description", "").strip()
                    abstraction = row.get("Abstraction", "").strip()
                    status = row.get("Status", "").strip()
                    likelihood = row.get("Likelihood Of Attack", "").strip()
                    severity = row.get("Typical Severity", "").strip()

                    # Parse example instances
                    example_instances = self._parse_example_instances(
                        row.get("Example Instances", "")
                    )

                    # Parse execution flow
                    execution_flow = self._parse_execution_flow(
                        row.get("Execution Flow", "")
                    )

                    # Parse related attack patterns
                    related_patterns = self._parse_related_patterns(
                        row.get("Related Attack Patterns", "")
                    )

                    # Parse related weaknesses (CWEs)
                    related_cwes = self._parse_related_weaknesses(
                        row.get("Related Weaknesses", "")
                    )

                    # Parse additional context fields
                    prerequisites = self._parse_prerequisites(
                        row.get("Prerequisites", "")
                    )
                    skills_required = self._parse_skills_required(
                        row.get("Skills Required", "")
                    )
                    resources_required = self._parse_resources_required(
                        row.get("Resources Required", "")
                    )
                    indicators = self._parse_indicators(row.get("Indicators", ""))
                    consequences = self._parse_consequences(row.get("Consequences", ""))
                    mitigations = self._parse_mitigations(row.get("Mitigations", ""))

                    # Parse taxonomy mappings
                    taxonomy_mappings = self._parse_taxonomy_mappings(
                        row.get("Taxonomy Mappings", "")
                    )
                    attack_ids = self._extract_attack_mappings(taxonomy_mappings)

                    # Log mapping information for debugging
                    if attack_ids:
                        logger.info(
                            f"Found {len(attack_ids)} ATT&CK mappings for CAPEC-{capec_id}: {attack_ids}"
                        )

                    # Store mappings for later processing
                    if attack_ids:
                        for attack_id in attack_ids:
                            attack_mappings.append(
                                {
                                    "capec_id": f"CAPEC-{capec_id}",
                                    "attack_id": attack_id,
                                }
                            )

                    # Insert into Neo4j
                    query = """
                    MERGE (p:CAPEC {capec_id: $capec_id})
                    ON CREATE SET 
                        p.name = $name,
                        p.description = $description,
                        p.abstraction = $abstraction,
                        p.status = $status,
                        p.likelihood = $likelihood,
                        p.severity = $severity,
                        p.execution_flow = $execution_flow,
                        p.prerequisites = $prerequisites,
                        p.skills_required = $skills_required,
                        p.resources_required = $resources_required,
                        p.indicators = $indicators,
                        p.consequences = $consequences,
                        p.mitigations = $mitigations,
                        p.example_instances = $example_instances,
                        p.created_at = datetime(),
                        p.updated_at = datetime()
                    ON MATCH SET 
                        p.name = $name,
                        p.description = $description,
                        p.abstraction = $abstraction,
                        p.status = $status,
                        p.likelihood = $likelihood,
                        p.severity = $severity,
                        p.execution_flow = $execution_flow,
                        p.prerequisites = $prerequisites,
                        p.skills_required = $skills_required,
                        p.resources_required = $resources_required,
                        p.indicators = $indicators,
                        p.consequences = $consequences,
                        p.mitigations = $mitigations,
                        p.example_instances = $example_instances,
                        p.updated_at = datetime()
                    """
                    params = {
                        "capec_id": f"CAPEC-{capec_id}",
                        "name": name,
                        "description": description,
                        "abstraction": abstraction,
                        "status": status,
                        "likelihood": likelihood,
                        "severity": severity,
                        "execution_flow": json.dumps(execution_flow),
                        "prerequisites": json.dumps(prerequisites),
                        "skills_required": json.dumps(skills_required),
                        "resources_required": json.dumps(resources_required),
                        "indicators": json.dumps(indicators),
                        "consequences": json.dumps(consequences),
                        "mitigations": json.dumps(mitigations),
                        "example_instances": json.dumps(example_instances),
                    }
                    self.neo4j.run_query(query, params)

                    # Create relationships to CWEs
                    for cwe_id in related_cwes:
                        query = """
                        MATCH (p:CAPEC {capec_id: $capec_id})
                        MERGE (c:CWE {cwe_id: $cwe_id})
                        ON CREATE SET 
                            c.created_at = datetime(),
                            c.updated_at = datetime()
                        ON MATCH SET 
                            c.updated_at = datetime()
                        MERGE (p)-[r:TARGETS_WEAKNESS]->(c)
                        ON CREATE SET r.created_at = datetime()
                        """
                        params = {"capec_id": f"CAPEC-{capec_id}", "cwe_id": cwe_id}
                        self.neo4j.run_query(query, params)

                    # Create relationships to other CAPEC patterns
                    for rel_pattern in related_patterns:
                        rel_id = rel_pattern.get("id", "")
                        rel_type = rel_pattern.get("type", "RELATED_TO")

                        if not rel_id:
                            continue

                        query = f"""
                        MATCH (p1:CAPEC {{capec_id: $capec_id}})
                        MERGE (p2:CAPEC {{capec_id: $related_id}})
                        ON CREATE SET 
                            p2.created_at = datetime(),
                            p2.updated_at = datetime()
                        ON MATCH SET 
                            p2.updated_at = datetime()
                        MERGE (p1)-[r:{rel_type}]->(p2)
                        ON CREATE SET r.created_at = datetime()
                        """
                        params = {
                            "capec_id": f"CAPEC-{capec_id}",
                            "related_id": f"CAPEC-{rel_id}",
                        }
                        self.neo4j.run_query(query, params)

                    patterns_count += 1
                    if patterns_count % 100 == 0:
                        logger.info(f"Processed {patterns_count} CAPEC patterns so far")

            logger.info(f"Processed {patterns_count} CAPEC patterns")

            # Create mappings between CAPEC and ATT&CK based on taxonomy mappings
            logger.info(
                "Creating mappings between CAPEC and ATT&CK based on taxonomy..."
            )
            if attack_mappings:
                mapping_count = 0

                for mapping in attack_mappings:
                    query = """
                    MATCH (p:CAPEC {capec_id: $capec_id})
                    MATCH (a:AttackTechnique {technique_id: $attack_id})
                    MERGE (p)-[r:MAPPED_TO]->(a)
                    ON CREATE SET r.created_at = datetime(), r.mapping_source = "taxonomy"
                    """

                    result = self.neo4j.run_query(query, mapping)
                    if result is not None:  # Check if the query was successful
                        mapping_count += 1

                logger.info(
                    f"Created {mapping_count} direct mappings between CAPEC and ATT&CK techniques"
                )
            else:
                logger.warning("No taxonomy mappings found for ATT&CK techniques")

            # Always create name-based mappings too, not just as a fallback
            logger.info("Creating additional name-based similarity mappings...")
            self._ensure_name_indexes()
            self._create_capec_attack_mappings()

        except Exception as e:
            logger.error(f"Error processing CAPEC data: {e}")
            logger.exception(e)  # This will log the full stack trace

        logger.info("Completed processing CAPEC data")

    def _create_capec_attack_mappings(self) -> None:
        """Create mappings between CAPEC and ATT&CK techniques using improved similarity."""
        try:
            # Get all CAPEC patterns
            capec_query = """
            MATCH (p:CAPEC)
            RETURN p.capec_id as id, p.name as name
            """
            capec_patterns = self.neo4j.run_query(capec_query)

            if not capec_patterns:
                logger.warning("No CAPEC patterns found for mapping to ATT&CK")
                return

            logger.info(
                f"Found {len(capec_patterns)} CAPEC patterns to map to ATT&CK techniques"
            )

            # Process each CAPEC pattern separately
            mappings_count = 0

            # Get all ATT&CK techniques first to avoid repeated queries
            attack_query = """
            MATCH (a:AttackTechnique)
            RETURN a.technique_id as id, a.name as name
            """
            attack_techniques = self.neo4j.run_query(attack_query)
            logger.info(
                f"Found {len(attack_techniques)} ATT&CK techniques for matching"
            )

            # Create a batch of relationships to process at once
            relationships = []

            for pattern in capec_patterns:
                if not pattern.get("name"):
                    continue

                capec_name = pattern["name"].lower()
                capec_words = set(re.findall(r"\b\w+\b", capec_name))

                # Find matching techniques
                for technique in attack_techniques:
                    if not technique.get("name"):
                        continue

                    technique_name = technique["name"].lower()
                    technique_words = set(re.findall(r"\b\w+\b", technique_name))

                    # Calculate word overlap (more flexible than substring)
                    common_words = capec_words.intersection(technique_words)
                    if len(common_words) >= 2:  # At least 2 common words
                        relationships.append(
                            {
                                "capec_id": pattern["id"],
                                "attack_id": technique["id"],
                                "common_words": ", ".join(common_words),
                            }
                        )

            # Create all relationships in batch
            if relationships:
                batch_query = """
                UNWIND $relationships as rel
                MATCH (p:CAPEC {capec_id: rel.capec_id})
                MATCH (a:AttackTechnique {technique_id: rel.attack_id})
                MERGE (p)-[r:SIMILAR_TO]->(a)
                ON CREATE SET 
                    r.created_at = datetime(), 
                    r.similarity = "name_match",
                    r.common_words = rel.common_words
                """
                result = self.neo4j.run_query(
                    batch_query, {"relationships": relationships}
                )
                mappings_count = len(relationships)

            logger.info(
                f"Created {mappings_count} SIMILAR_TO relationships between CAPEC and ATT&CK techniques"
            )

        except Exception as e:
            logger.error(f"Error creating CAPEC to ATT&CK mappings: {e}")

    def _ensure_name_indexes(self) -> None:
        """Ensure indexes exist on name properties for better performance."""
        try:
            # Create index on AttackTechnique.name if it doesn't exist
            index_attack_query = """
            CREATE INDEX IF NOT EXISTS FOR (t:AttackTechnique) ON (t.name)
            """
            self.neo4j.run_query(index_attack_query)

            # Create index on CAPEC.name if it doesn't exist
            index_capec_query = """
            CREATE INDEX IF NOT EXISTS FOR (p:CAPEC) ON (p.name)
            """
            self.neo4j.run_query(index_capec_query)

            logger.info("Ensured indexes exist on name properties")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")

    def _parse_prerequisites(self, prerequisites_str: str) -> List[str]:
        """Parse prerequisites string into structured data.

        Args:
            prerequisites_str: String containing prerequisites information

        Returns:
            List of prerequisites
        """
        if (
            not prerequisites_str
            or pd.isna(prerequisites_str)
            or prerequisites_str == ""
        ):
            return []

        prerequisites = []
        # Split by double colons (::) for separate prerequisites
        if "::" in prerequisites_str:
            prerequisite_parts = prerequisites_str.split("::")
            for part in prerequisite_parts:
                part = part.strip()
                if part and not part.isspace():
                    prerequisites.append(part)
        else:
            # If no double colons, treat the whole string as one prerequisite
            prerequisites.append(prerequisites_str.strip())

        return prerequisites

    def _parse_skills_required(self, skills_str: str) -> List[Dict]:
        """Parse skills required string into structured data.

        Args:
            skills_str: String containing skills required information

        Returns:
            List of dictionaries with skill information
        """
        if not skills_str or pd.isna(skills_str) or skills_str == "":
            return []

        skills = []
        # Look for pattern ::SKILL:description:LEVEL:level::
        skill_pattern = r"::SKILL:([^:]+):LEVEL:([^:]+)::"

        for match in re.finditer(skill_pattern, skills_str):
            description = match.group(1).strip()
            level = match.group(2).strip()

            skills.append({"description": description, "level": level})

        # If no matches found, try alternate format
        if not skills and "::" in skills_str:
            skill_parts = skills_str.split("::")
            for part in skill_parts:
                part = part.strip()
                if part and not part.isspace():
                    skills.append({"description": part, "level": "Unknown"})
        elif not skills and skills_str:
            # If no pattern matches but string exists, add as is
            skills.append({"description": skills_str.strip(), "level": "Unknown"})

        return skills

    def _parse_resources_required(self, resources_str: str) -> List[str]:
        """Parse resources required string into structured data.

        Args:
            resources_str: String containing resources required information

        Returns:
            List of required resources
        """
        if not resources_str or pd.isna(resources_str) or resources_str == "":
            return []

        resources = []
        # Split by double colons (::) for separate resources
        if "::" in resources_str:
            resource_parts = resources_str.split("::")
            for part in resource_parts:
                part = part.strip()
                if part and not part.isspace():
                    resources.append(part)
        else:
            # If no double colons, treat the whole string as one resource
            resources.append(resources_str.strip())

        return resources

    def _parse_indicators(self, indicators_str: str) -> List[str]:
        """Parse indicators string into structured data.

        Args:
            indicators_str: String containing indicators information

        Returns:
            List of indicators
        """
        if not indicators_str or pd.isna(indicators_str) or indicators_str == "":
            return []

        indicators = []
        # Split by double colons (::) for separate indicators
        if "::" in indicators_str:
            indicator_parts = indicators_str.split("::")
            for part in indicator_parts:
                part = part.strip()
                if part and not part.isspace():
                    indicators.append(part)
        else:
            # If no double colons, treat the whole string as one indicator
            indicators.append(indicators_str.strip())

        return indicators

    def _parse_consequences(self, consequences_str: str) -> List[Dict]:
        """Parse consequences string into structured data.

        Args:
            consequences_str: String containing consequences information

        Returns:
            List of dictionaries with consequence information
        """
        if not consequences_str or pd.isna(consequences_str) or consequences_str == "":
            return []

        consequences = []
        # Look for patterns like ::SCOPE:X:SCOPE:Y:TECHNICAL IMPACT:Z::
        scope_pattern = (
            r"::(?:SCOPE:([^:]+):)*TECHNICAL IMPACT:([^:]+)(?::NOTE:([^:]+))?::"
        )

        for match in re.finditer(scope_pattern, consequences_str):
            # Try to extract all scope parts
            scopes = []
            if match.group(1):
                scopes.append(match.group(1).strip())

            # Handle multiple scopes by looking at the raw match string
            scope_parts = re.findall(r"SCOPE:([^:]+):", match.group(0))
            if scope_parts:
                scopes = [scope.strip() for scope in scope_parts]

            technical_impact = match.group(2).strip() if match.group(2) else ""
            note = match.group(3).strip() if match.group(3) else ""

            consequences.append(
                {"scopes": scopes, "technical_impact": technical_impact, "note": note}
            )

        # If no structured consequences found, try to parse as plain text
        if not consequences and "::" in consequences_str:
            conseq_parts = consequences_str.split("::")
            for part in conseq_parts:
                part = part.strip()
                if part and not part.isspace():
                    consequences.append(
                        {"text": part, "scopes": [], "technical_impact": "Unknown"}
                    )
        elif not consequences and consequences_str:
            consequences.append(
                {
                    "text": consequences_str.strip(),
                    "scopes": [],
                    "technical_impact": "Unknown",
                }
            )

        return consequences

    def _parse_mitigations(self, mitigations_str: str) -> List[str]:
        """Parse mitigations string into structured data.

        Args:
            mitigations_str: String containing mitigations information

        Returns:
            List of mitigations
        """
        if not mitigations_str or pd.isna(mitigations_str) or mitigations_str == "":
            return []

        mitigations = []
        # Split by double colons (::) for separate mitigations
        if "::" in mitigations_str:
            mitigation_parts = mitigations_str.split("::")
            for part in mitigation_parts:
                part = part.strip()
                if part and not part.isspace():
                    mitigations.append(part)
        else:
            # If no double colons, treat the whole string as one mitigation
            mitigations.append(mitigations_str.strip())

        return mitigations

    def _parse_execution_flow(self, execution_flow_str: str) -> List[Dict]:
        """Parse execution flow string into structured data.

        Args:
            execution_flow_str: String containing execution flow information

        Returns:
            List of dictionaries with parsed execution flow steps
        """
        if (
            not execution_flow_str
            or pd.isna(execution_flow_str)
            or execution_flow_str == ""
        ):
            return []

        steps = []
        step_pattern = (
            r"::STEP:(\d+):PHASE:([^:]+):DESCRIPTION:([^:]+)(?::TECHNIQUE:([^:]+))?"
        )

        for match in re.finditer(step_pattern, str(execution_flow_str)):
            step_num = match.group(1)
            phase = match.group(2)
            description = match.group(3)
            techniques = []

            if match.group(4):
                techniques = [
                    tech.strip() for tech in match.group(4).split(":TECHNIQUE:")
                ]

            steps.append(
                {
                    "step_number": step_num,
                    "phase": phase,
                    "description": description,
                    "techniques": techniques,
                }
            )

        return steps

    def _parse_related_patterns(self, related_patterns_str: str) -> List[Dict]:
        """Parse related attack patterns string into structured data.

        Args:
            related_patterns_str: String containing related patterns information

        Returns:
            List of dictionaries with parsed related patterns
        """
        if (
            not related_patterns_str
            or pd.isna(related_patterns_str)
            or related_patterns_str == ""
        ):
            return []

        patterns = []
        pattern_regex = r"::NATURE:([^:]+):CAPEC ID:(\d+)"

        for match in re.finditer(pattern_regex, related_patterns_str):
            relation_type = match.group(1)
            capec_id = match.group(2)

            # Map relation type to relationship type
            rel_type = "RELATED_TO"
            if relation_type == "ChildOf":
                rel_type = "CHILD_OF"
            elif relation_type == "ParentOf":
                rel_type = "PARENT_OF"
            elif relation_type == "CanFollow":
                rel_type = "CAN_FOLLOW"
            elif relation_type == "CanPrecede":
                rel_type = "CAN_PRECEDE"

            patterns.append({"id": capec_id, "type": rel_type})

        return patterns

    def _parse_related_weaknesses(self, related_weaknesses_str: str) -> List[str]:
        """Parse related weaknesses string into CWE IDs.

        Args:
            related_weaknesses_str: String containing related weaknesses information

        Returns:
            List of CWE IDs
        """
        if (
            not related_weaknesses_str
            or pd.isna(related_weaknesses_str)
            or related_weaknesses_str == ""
        ):
            return []

        cwe_ids = []
        # The format appears to be space or double-colon separated IDs
        if "::" in related_weaknesses_str:
            # Parse pattern like "::276::285::434::693::"
            cwe_pattern = r"::(\d+)::"
            cwe_ids = [
                f"CWE-{match.group(1)}"
                for match in re.finditer(cwe_pattern, related_weaknesses_str)
            ]
        else:
            # Try to parse space-separated IDs
            for token in related_weaknesses_str.split():
                if token.isdigit():
                    cwe_ids.append(f"CWE-{token}")

        return cwe_ids

    def update(self) -> None:
        """Update CAPEC data in the database."""
        self.process_capec_data()


class CVEConnector:
    """Handles processing of CVE data from CSV."""

    def __init__(self, neo4j_connector: Neo4jConnector):
        """Initialize the CVE connector.

        Args:
            neo4j_connector: An initialized Neo4j connector
        """
        self.neo4j = neo4j_connector
        self.cve_csv_path = "data/nvd_vulnerabilities_merged.csv"

    def process_cve_data(self) -> None:
        """Process and insert CVE data from CSV into Neo4j with optimizations."""
        logger.info(f"Processing CVE data from {self.cve_csv_path}...")

        try:
            cve_count = 0
            cwe_cache = set()

            # Use pandas with chunking to handle large files efficiently
            chunk_size = 5000

            for cve_df_chunk in pd.read_csv(self.cve_csv_path, chunksize=chunk_size):
                logger.info(f"Processing chunk of {len(cve_df_chunk)} CVE records")

                # Prepare batch for better performance
                cve_batch = []
                cwe_relationships_batch = []

                for _, row in cve_df_chunk.iterrows():
                    cve_id = row.get("id")
                    if (
                        not cve_id
                        or not isinstance(cve_id, str)
                        or not cve_id.startswith("CVE-")
                    ):
                        continue

                    # Extract basic information
                    published_date = row.get("published", "")
                    last_modified_date = row.get("lastModified", "")
                    description = row.get("description_en", "")

                    # Extract CVSS scores
                    cvss_v2_data = {
                        "source": row.get("cvss2_source", ""),
                        "type": row.get("cvss2_type", ""),
                        "vectorString": row.get("cvss2_vectorString", ""),
                        "baseScore": (
                            float(row.get("cvss2_baseScore"))
                            if pd.notna(row.get("cvss2_baseScore"))
                            else 0.0
                        ),
                        "baseSeverity": row.get("cvss2_baseSeverity", ""),
                    }

                    cvss_v3_data = {
                        "source": row.get("cvss31_source", ""),
                        "type": row.get("cvss31_type", ""),
                        "vectorString": row.get("cvss31_vectorString", ""),
                        "baseScore": (
                            float(row.get("cvss31_baseScore"))
                            if pd.notna(row.get("cvss31_baseScore"))
                            else 0.0
                        ),
                        "baseSeverity": row.get("cvss31_baseSeverity", ""),
                    }

                    # Extract references
                    references = self._parse_references(row.get("references", ""))

                    # Extract configurations (affected products)
                    configurations = self._parse_configurations(
                        row.get("configurations", "")
                    )

                    # Extract weaknesses (CWEs)
                    cwe_ids = self._parse_weaknesses(row.get("weaknesses", ""))

                    # Add to batch
                    cve_batch.append(
                        {
                            "cve_id": cve_id,
                            "description": description,
                            "published_date": published_date,
                            "last_modified_date": last_modified_date,
                            "cvss_v2_base_score": cvss_v2_data["baseScore"],
                            "cvss_v3_base_score": cvss_v3_data["baseScore"],
                            "cvss_v2": json.dumps(cvss_v2_data),
                            "cvss_v3": json.dumps(cvss_v3_data),
                            "references": json.dumps(references),
                            "configurations": json.dumps(configurations),
                        }
                    )

                    # Add CWE relationships to batch
                    for cwe_id in cwe_ids:
                        cwe_cache.add(cwe_id)
                        cwe_relationships_batch.append(
                            {"cve_id": cve_id, "cwe_id": cwe_id}
                        )

                    cve_count += 1

                # Process the entire batch at once
                if cve_batch:
                    # Insert CVEs in batch
                    batch_query = """
                    UNWIND $cve_list as cve
                    MERGE (v:CVE {cve_id: cve.cve_id})
                    ON CREATE SET 
                        v.description = cve.description,
                        v.published_date = datetime(cve.published_date),
                        v.last_modified_date = datetime(cve.last_modified_date),
                        v.cvss_v2_base_score = cve.cvss_v2_base_score,
                        v.cvss_v3_base_score = cve.cvss_v3_base_score,
                        v.cvss_v2 = cve.cvss_v2,
                        v.cvss_v3 = cve.cvss_v3,
                        v.references = cve.references,
                        v.configurations = cve.configurations,
                        v.created_at = datetime(),
                        v.updated_at = datetime()
                    ON MATCH SET 
                        v.description = cve.description,
                        v.published_date = datetime(cve.published_date),
                        v.last_modified_date = datetime(cve.last_modified_date),
                        v.cvss_v2_base_score = cve.cvss_v2_base_score,
                        v.cvss_v3_base_score = cve.cvss_v3_base_score,
                        v.cvss_v2 = cve.cvss_v2,
                        v.cvss_v3 = cve.cvss_v3,
                        v.references = cve.references,
                        v.configurations = cve.configurations,
                        v.updated_at = datetime()
                    """
                    self.neo4j.run_query(batch_query, {"cve_list": cve_batch})

                    # Create CWEs first to avoid missing node errors
                    if cwe_cache:
                        cwe_batch_query = """
                        UNWIND $cwe_list as cwe_id
                        MERGE (c:CWE {cwe_id: cwe_id})
                        ON CREATE SET 
                            c.created_at = datetime(),
                            c.updated_at = datetime()
                        ON MATCH SET 
                            c.updated_at = datetime()
                        """
                        self.neo4j.run_query(
                            cwe_batch_query, {"cwe_list": list(cwe_cache)}
                        )
                        cwe_cache.clear()

                    # Create relationships in batch
                    if cwe_relationships_batch:
                        rel_batch_query = """
                        UNWIND $relationships as rel
                        MATCH (v:CVE {cve_id: rel.cve_id})
                        MATCH (c:CWE {cwe_id: rel.cwe_id})
                        MERGE (v)-[r:EXHIBITS_WEAKNESS]->(c)
                        ON CREATE SET r.created_at = datetime()
                        """
                        self.neo4j.run_query(
                            rel_batch_query, {"relationships": cwe_relationships_batch}
                        )

                logger.info(f"Processed {cve_count} CVEs so far")

            logger.info(f"Completed processing {cve_count} CVE records")

        except Exception as e:
            logger.error(f"Error processing CVE data: {e}")

    def _parse_references(self, references_str: str) -> List[Dict]:
        """Parse references string into structured data.

        Args:
            references_str: String containing references information

        Returns:
            List of dictionaries with parsed references
        """
        if not references_str or references_str == "NaN" or pd.isna(references_str):
            return []

        references = []
        # Split reference entries (they appear to be separated by ||)
        if "||" in references_str:
            for ref_entry in references_str.split("||"):
                ref_parts = ref_entry.split(";")
                ref_data = {}

                for part in ref_parts:
                    if part.startswith("url="):
                        ref_data["url"] = part[4:]
                    elif part.startswith("source="):
                        ref_data["source"] = part[7:]
                    elif part.startswith("tags="):
                        tags_str = part[5:]
                        if tags_str and tags_str != "None":
                            ref_data["tags"] = tags_str.split(",")

                if "url" in ref_data:
                    references.append(ref_data)

        return references

    def _parse_configurations(self, config_str: str) -> List[Dict]:
        """Parse configurations string into structured data.

        Args:
            config_str: String containing configurations information

        Returns:
            List of dictionaries with parsed configurations
        """
        if not config_str or config_str == "NaN" or pd.isna(config_str):
            return []

        configurations = []
        # Split configuration entries (they appear to be separated by ||)
        if "||" in config_str:
            for config_entry in config_str.split("||"):
                if config_entry.startswith("criteria="):
                    cpe_parts = config_entry.split(";")
                    cpe_data = {}

                    for part in cpe_parts:
                        if part.startswith("criteria="):
                            cpe_data["criteria"] = part[9:]
                        elif part.startswith("vulnerable="):
                            cpe_data["vulnerable"] = part[11:].lower() == "true"
                        elif part.startswith("operator="):
                            cpe_data["operator"] = part[9:]

                    if "criteria" in cpe_data:
                        configurations.append(cpe_data)

        return configurations

    def _parse_weaknesses(self, weaknesses_str: str) -> List[str]:
        """Parse weaknesses string into CWE IDs.

        Args:
            weaknesses_str: String containing weaknesses information

        Returns:
            List of CWE IDs
        """
        if not weaknesses_str or weaknesses_str == "NaN" or pd.isna(weaknesses_str):
            return []

        cwe_ids = []
        # The format appears to be something like "NVD-CWE-Other:nvd@nist.gov:Primary"
        # We'll look for the common format "CWE-\d+"
        cwe_pattern = r"CWE-(\d+)"
        cwe_matches = re.findall(cwe_pattern, weaknesses_str)

        if cwe_matches:
            cwe_ids = [f"CWE-{cwe}" for cwe in cwe_matches]
        else:
            # If no standard CWE ID is found, check if it's using some other format
            # For now, we'll just log this case
            logger.debug(f"Non-standard CWE format: {weaknesses_str}")

        return cwe_ids

    def update(self) -> None:
        """Update CVE data in the database."""
        self.process_cve_data()


class KEVConnector:
    """Handles processing of CISA Known Exploited Vulnerabilities (KEV) data."""

    def __init__(self, neo4j_connector: Neo4jConnector):
        """Initialize the KEV connector.

        Args:
            neo4j_connector: An initialized Neo4j connector
        """
        self.neo4j = neo4j_connector
        self.kev_cve_cwe_path = "data/kev_cve_cwe.csv"
        self.kev_cve_attack_path = "data/kev_cve_attack.csv"

    def process_kev_data(self) -> None:
        """Process and insert KEV data into Neo4j."""
        logger.info("Processing CISA KEV data...")

        # First process KEV CVE-CWE data
        self._process_kev_cve_cwe_data()

        # Then process KEV CVE-ATT&CK data
        self._process_kev_cve_attack_data()

        logger.info("Completed processing KEV data")

    def _process_kev_cve_cwe_data(self) -> None:
        """Process KEV CVE-CWE data and integrate into Neo4j."""
        try:
            logger.info(f"Processing KEV CVE-CWE data from {self.kev_cve_cwe_path}...")

            # Read the KEV CVE-CWE CSV
            kev_df = pd.read_csv(self.kev_cve_cwe_path)
            logger.info(f"Loaded {len(kev_df)} rows from KEV CVE-CWE CSV")

            # Track statistics
            cve_updated = 0
            cwe_relationships = 0

            # Process each KEV entry
            for _, row in kev_df.iterrows():
                cve_id = row.get("cveID")

                if (
                    not cve_id
                    or not isinstance(cve_id, str)
                    or not cve_id.startswith("CVE-")
                ):
                    continue

                # Extract KEV-specific information
                vendor_project = row.get("vendorProject", "")
                product = row.get("product", "")
                vulnerability_name = row.get("vulnerabilityName", "")
                short_description = row.get("shortDescription", "")
                required_action = row.get("requiredAction", "")
                known_ransomware = row.get("knownRansomwareCampaignUse", "Unknown")
                notes = row.get("notes", "")

                # Convert dates to proper format
                date_added = self._convert_date_format(row.get("dateAdded", ""))
                due_date = self._convert_date_format(row.get("dueDate", ""))

                # Prepare query based on available dates
                if date_added and due_date:
                    query = """
                    MERGE (v:CVE {cve_id: $cve_id})
                    ON CREATE SET 
                        v.created_at = datetime(),
                        v.updated_at = datetime()
                    ON MATCH SET 
                        v.updated_at = datetime()
                    SET 
                        v.is_kev = true,
                        v.kev_date_added = date($date_added),
                        v.kev_due_date = date($due_date),
                        v.kev_vendor_project = $vendor_project,
                        v.kev_product = $product,
                        v.kev_vulnerability_name = $vulnerability_name,
                        v.kev_short_description = $short_description,
                        v.kev_required_action = $required_action,
                        v.kev_known_ransomware = $known_ransomware,
                        v.kev_notes = $notes
                    """
                elif date_added:
                    query = """
                    MERGE (v:CVE {cve_id: $cve_id})
                    ON CREATE SET 
                        v.created_at = datetime(),
                        v.updated_at = datetime()
                    ON MATCH SET 
                        v.updated_at = datetime()
                    SET 
                        v.is_kev = true,
                        v.kev_date_added = date($date_added),
                        v.kev_vendor_project = $vendor_project,
                        v.kev_product = $product,
                        v.kev_vulnerability_name = $vulnerability_name,
                        v.kev_short_description = $short_description,
                        v.kev_required_action = $required_action,
                        v.kev_known_ransomware = $known_ransomware,
                        v.kev_notes = $notes
                    """
                elif due_date:
                    query = """
                    MERGE (v:CVE {cve_id: $cve_id})
                    ON CREATE SET 
                        v.created_at = datetime(),
                        v.updated_at = datetime()
                    ON MATCH SET 
                        v.updated_at = datetime()
                    SET 
                        v.is_kev = true,
                        v.kev_due_date = date($due_date),
                        v.kev_vendor_project = $vendor_project,
                        v.kev_product = $product,
                        v.kev_vulnerability_name = $vulnerability_name,
                        v.kev_short_description = $short_description,
                        v.kev_required_action = $required_action,
                        v.kev_known_ransomware = $known_ransomware,
                        v.kev_notes = $notes
                    """
                else:
                    query = """
                    MERGE (v:CVE {cve_id: $cve_id})
                    ON CREATE SET 
                        v.created_at = datetime(),
                        v.updated_at = datetime()
                    ON MATCH SET 
                        v.updated_at = datetime()
                    SET 
                        v.is_kev = true,
                        v.kev_vendor_project = $vendor_project,
                        v.kev_product = $product,
                        v.kev_vulnerability_name = $vulnerability_name,
                        v.kev_short_description = $short_description,
                        v.kev_required_action = $required_action,
                        v.kev_known_ransomware = $known_ransomware,
                        v.kev_notes = $notes
                    """

                params = {
                    "cve_id": cve_id,
                    "date_added": date_added,
                    "due_date": due_date,
                    "vendor_project": vendor_project,
                    "product": product,
                    "vulnerability_name": vulnerability_name,
                    "short_description": short_description,
                    "required_action": required_action,
                    "known_ransomware": known_ransomware == "Known",
                    "notes": notes,
                }

                self.neo4j.run_query(query, params)
                cve_updated += 1

                # Process CWE relationships
                cwe_data = row.get("cwes", "")
                if cwe_data and isinstance(cwe_data, str):
                    # Parse CWE IDs - they might be comma-separated
                    cwe_ids = []

                    # Handle multiple CWEs (comma-separated or in various formats)
                    if "," in cwe_data:
                        for cwe_part in cwe_data.split(","):
                            cwe_match = re.search(r"CWE-(\d+)", cwe_part.strip())
                            if cwe_match:
                                cwe_ids.append(f"CWE-{cwe_match.group(1)}")
                    else:
                        # Look for all CWE mentions in the string
                        cwe_matches = re.findall(r"CWE-(\d+)", cwe_data)
                        cwe_ids = [f"CWE-{match}" for match in cwe_matches]

                    # Create relationships for each CWE
                    for cwe_id in cwe_ids:
                        # Create or update the CWE node and relationship
                        cwe_query = """
                        MERGE (c:CWE {cwe_id: $cwe_id})
                        ON CREATE SET 
                            c.created_at = datetime(),
                            c.updated_at = datetime()
                        ON MATCH SET 
                            c.updated_at = datetime()
                        
                        WITH c
                        MATCH (v:CVE {cve_id: $cve_id})
                        MERGE (v)-[r:EXHIBITS_WEAKNESS]->(c)
                        ON CREATE SET 
                            r.created_at = datetime(),
                            r.source = "KEV"
                        """

                        self.neo4j.run_query(
                            cwe_query, {"cve_id": cve_id, "cwe_id": cwe_id}
                        )
                        cwe_relationships += 1

                # Log progress periodically
                if cve_updated % 100 == 0:
                    logger.info(f"Processed {cve_updated} KEV CVEs so far")

            logger.info(f"Updated {cve_updated} CVE nodes with KEV data")
            logger.info(
                f"Created {cwe_relationships} CVE-CWE relationships from KEV data"
            )

        except Exception as e:
            logger.error(f"Error processing KEV CVE-CWE data: {e}", exc_info=True)

    def _process_kev_cve_attack_data(self) -> None:
        """Process KEV CVE-ATT&CK mapping data and integrate into Neo4j."""
        try:
            logger.info(
                f"Processing KEV CVE-ATT&CK data from {self.kev_cve_attack_path}..."
            )

            # Read the KEV CVE-ATT&CK CSV
            kev_attack_df = pd.read_csv(self.kev_cve_attack_path)
            logger.info(f"Loaded {len(kev_attack_df)} rows from KEV CVE-ATT&CK CSV")

            # Track statistics
            attack_relationships = 0

            # Process each mapping entry
            for _, row in kev_attack_df.iterrows():
                capability_id = row.get("capability_id")

                # Ensure capability_id is a CVE ID
                if (
                    not capability_id
                    or not isinstance(capability_id, str)
                    or not capability_id.startswith("CVE-")
                ):
                    continue

                # Extract the ATT&CK technique ID
                attack_id = row.get("attack_object_id")
                if not attack_id or not isinstance(attack_id, str):
                    continue

                # Extract additional mapping information
                mapping_type = row.get("mapping_type", "")
                capability_group = row.get("capability_group", "")
                capability_description = row.get("capability_description", "")
                attack_name = row.get("attack_object_name", "")

                # Convert date formats - handle MM/DD/YYYY to YYYY-MM-DD
                creation_date = self._convert_date_format(row.get("creation_date", ""))
                update_date = self._convert_date_format(row.get("update_date", ""))

                # Create the relationship between CVE and ATT&CK technique
                # If dates are empty, don't try to use the date() function
                if creation_date and update_date:
                    query = """
                    MATCH (v:CVE {cve_id: $cve_id})
                    MATCH (a:AttackTechnique {technique_id: $attack_id})
                    MERGE (v)-[r:ENABLES]->(a)
                    ON CREATE SET 
                        r.created_at = datetime(),
                        r.source = "KEV",
                        r.mapping_type = $mapping_type,
                        r.capability_group = $capability_group,
                        r.capability_description = $capability_description,
                        r.kev_creation_date = date($creation_date),
                        r.kev_update_date = date($update_date)
                    ON MATCH SET
                        r.updated_at = datetime(),
                        r.mapping_type = $mapping_type,
                        r.capability_group = $capability_group,
                        r.capability_description = $capability_description,
                        r.kev_creation_date = date($creation_date),
                        r.kev_update_date = date($update_date)
                    """
                elif creation_date:
                    query = """
                    MATCH (v:CVE {cve_id: $cve_id})
                    MATCH (a:AttackTechnique {technique_id: $attack_id})
                    MERGE (v)-[r:ENABLES]->(a)
                    ON CREATE SET 
                        r.created_at = datetime(),
                        r.source = "KEV",
                        r.mapping_type = $mapping_type,
                        r.capability_group = $capability_group,
                        r.capability_description = $capability_description,
                        r.kev_creation_date = date($creation_date)
                    ON MATCH SET
                        r.updated_at = datetime(),
                        r.mapping_type = $mapping_type,
                        r.capability_group = $capability_group,
                        r.capability_description = $capability_description,
                        r.kev_creation_date = date($creation_date)
                    """
                else:
                    query = """
                    MATCH (v:CVE {cve_id: $cve_id})
                    MATCH (a:AttackTechnique {technique_id: $attack_id})
                    MERGE (v)-[r:ENABLES]->(a)
                    ON CREATE SET 
                        r.created_at = datetime(),
                        r.source = "KEV",
                        r.mapping_type = $mapping_type,
                        r.capability_group = $capability_group,
                        r.capability_description = $capability_description
                    ON MATCH SET
                        r.updated_at = datetime(),
                        r.mapping_type = $mapping_type,
                        r.capability_group = $capability_group,
                        r.capability_description = $capability_description
                    """

                params = {
                    "cve_id": capability_id,
                    "attack_id": attack_id,
                    "mapping_type": mapping_type,
                    "capability_group": capability_group,
                    "capability_description": capability_description,
                    "creation_date": creation_date,
                    "update_date": update_date,
                }

                self.neo4j.run_query(query, params)
                attack_relationships += 1

                # Log progress periodically
                if attack_relationships % 100 == 0:
                    logger.info(
                        f"Created {attack_relationships} CVE-ATT&CK relationships so far"
                    )

            logger.info(
                f"Created {attack_relationships} CVE-ATT&CK relationships from KEV data"
            )

        except Exception as e:
            logger.error(f"Error processing KEV CVE-ATT&CK data: {e}", exc_info=True)

    def _convert_date_format(self, date_str: str) -> str:
        """Convert date strings to YYYY-MM-DD format expected by Neo4j.

        Handles both MM/DD/YYYY and already-correct YYYY-MM-DD formats.

        Args:
            date_str: Date string in either MM/DD/YYYY or YYYY-MM-DD format

        Returns:
            Date string in YYYY-MM-DD format, or empty string if input is invalid
        """
        if not date_str or pd.isna(date_str):
            return ""

        # If it's already in YYYY-MM-DD format, just return it
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str

        try:
            # Try to parse as MM/DD/YYYY
            date_obj = datetime.strptime(date_str, "%m/%d/%Y")
            # Return in ISO format
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            try:
                # Try alternative formats that might be in the data
                # Check for M/D/YYYY format (single digits for month/day)
                if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", date_str):
                    parts = date_str.split("/")
                    month = int(parts[0])
                    day = int(parts[1])
                    year = int(parts[2])
                    return f"{year:04d}-{month:02d}-{day:02d}"

                # Check for other possible formats here if needed

                logger.warning(
                    f"Unrecognized date format, attempting flexible parsing: {date_str}"
                )
                # Last resort - try dateutil's flexible parser
                from dateutil import parser

                date_obj = parser.parse(date_str)
                return date_obj.strftime("%Y-%m-%d")

            except Exception:
                # If all parsing attempts fail, log and return empty string
                logger.warning(f"Could not parse date: {date_str}")
                return ""

    def update(self) -> None:
        """Update KEV data in the database."""
        self.process_kev_data()


class IntegrationManager:
    """Manages the integration process for all data sources."""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize the integration manager.

        Args:
            neo4j_uri: URI for the Neo4j database
            neo4j_user: Username for the Neo4j database
            neo4j_password: Password for the Neo4j database
        """
        self.neo4j = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
        if not self.neo4j.connect():
            raise ConnectionError("Failed to connect to Neo4j database")

        self.neo4j.create_constraints_and_indexes()

        # Initialize connectors
        self.attack_connector = MitreAttackConnector(self.neo4j)
        self.d3fend_connector = D3FENDConnector(self.neo4j)
        self.engage_connector = MitreEngageConnector(self.neo4j)
        self.capec_connector = CAPECConnector(self.neo4j)
        self.cve_connector = CVEConnector(self.neo4j)
        self.kev_connector = KEVConnector(self.neo4j)

    def run_full_update(self) -> None:
        """Run a full update of all data sources."""
        logger.info("Starting full update of all data sources...")

        # Create the data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Run updates sequentially to avoid overwhelming database
            # Update ATT&CK (core framework)
            logger.info("Updating MITRE ATT&CK...")
            self.attack_connector.update()

            # Update D3FEND
            logger.info("Updating D3FEND...")
            self.d3fend_connector.update()

            # Update Engage
            logger.info("Updating MITRE Engage...")
            self.engage_connector.update()

            # Update CAPEC
            logger.info("Updating CAPEC...")
            self.capec_connector.update()

            # Update CVE
            logger.info("Updating CVE...")
            self.cve_connector.update()

            # Update KEV
            logger.info("Updating CISA KEV data...")
            self.kev_connector.update()

        logger.info("Full update completed")

    def schedule_updates(self) -> None:
        """Schedule regular updates for all data sources."""
        # ATT&CK updates - weekly
        schedule.every().monday.at("01:00").do(self.attack_connector.update)

        # D3FEND updates - weekly
        schedule.every().monday.at("02:00").do(self.d3fend_connector.update)

        # Engage updates - weekly
        schedule.every().monday.at("03:00").do(self.engage_connector.update)

        # CAPEC updates - monthly
        schedule.every(30).days.at("05:00").do(self.capec_connector.update)

        # CVE updates - weekly
        schedule.every().monday.at("06:00").do(self.cve_connector.update)

        # KEV updates - weekly
        schedule.every().monday.at("04:00").do(self.kev_connector.update)

        logger.info("Update schedules set")

    def run_scheduler(self) -> None:
        """Run the scheduler loop."""
        logger.info("Starting scheduler...")
        while True:
            schedule.run_pending()
            time.sleep(60)

    def verify_source_integrity(self, url: str, expected_hash: str = None) -> bool:
        """
        Verify the integrity of a data source

        Args:
            url: URL of the data source
            expected_hash: Expected SHA-256 hash, if available

        Returns:
            Whether verification was successful
        """
        try:
            # Download the file
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content = response.content

            # Calculate hash
            calculated_hash = hashlib.sha256(content).hexdigest()

            # If no expected hash provided, just log and return True
            if not expected_hash:
                logger.info(f"Calculated SHA-256 hash for {url}: {calculated_hash}")
                return True

            # Compare with expected hash
            if calculated_hash == expected_hash:
                logger.info(f"Integrity verification successful for {url}")
                return True
            else:
                logger.error(f"Integrity verification failed for {url}")
                logger.error(f"Expected hash: {expected_hash}")
                logger.error(f"Calculated hash: {calculated_hash}")
                return False

        except Exception as e:
            logger.error(f"Error during source integrity verification: {e}")
            return False

    def _track_data_transformation(
        self,
        source_id: str,
        transformation_type: str,
        original_format: str,
        output_format: str,
        record_count: int,
    ) -> None:
        """
        Track data transformation for provenance

        Args:
            source_id: Identifier for the data source
            transformation_type: Type of transformation performed
            original_format: Original data format
            output_format: Output data format
            record_count: Number of records processed
        """
        transformation_record = {
            "source_id": source_id,
            "transformation_type": transformation_type,
            "original_format": original_format,
            "output_format": output_format,
            "record_count": record_count,
            "timestamp": datetime.utcnow().isoformat(),
            "process_id": os.getpid(),
        }

        # Save to transformations log
        os.makedirs("data/provenance", exist_ok=True)

        with open("data/provenance/transformations.jsonl", "a") as f:
            f.write(json.dumps(transformation_record) + "\n")

        # If Neo4j is available, also save there
        if self.neo4j and self.neo4j.driver:
            query = """
            CREATE (t:DataTransformation {
                source_id: $source_id,
                transformation_type: $transformation_type,
                original_format: $original_format,
                output_format: $output_format,
                record_count: $record_count,
                timestamp: datetime($timestamp),
                process_id: $process_id
            })
            """
            params = {
                "source_id": source_id,
                "transformation_type": transformation_type,
                "original_format": original_format,
                "output_format": output_format,
                "record_count": record_count,
                "timestamp": transformation_record["timestamp"],
                "process_id": str(transformation_record["process_id"]),
            }

            try:
                self.neo4j.run_query(query, params)
            except Exception as e:
                logger.error(f"Error saving transformation record to Neo4j: {e}")

    def close(self) -> None:
        """Close all connections."""
        self.neo4j.close()
        logger.info("All connections closed")


def update_attack_framework(self) -> bool:
    """
    Update ATT&CK framework data with optimized incremental updates

    Returns:
        Whether update was successful
    """
    logger.info("Updating ATT&CK framework data...")

    # Check if attack-last-update.json exists
    last_update_file = os.path.join("data", "attack-last-update.json")
    last_update_info = {}

    if os.path.exists(last_update_file):
        try:
            with open(last_update_file, "r") as f:
                last_update_info = json.load(f)
        except Exception as e:
            logger.error(f"Error reading last update info: {e}")

    # Get the last modified timestamp from the server
    try:
        head_response = requests.head(
            "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
        )
        head_response.raise_for_status()

        remote_last_modified = head_response.headers.get("Last-Modified", "")

        # If unchanged since last update, skip downloading
        if remote_last_modified and remote_last_modified == last_update_info.get(
            "last_modified"
        ):
            logger.info("ATT&CK data unchanged since last update, skipping download")
            return True

        # Proceed with download
        attack_data = self.attack_connector.fetch_attack_data()
        if not attack_data:
            logger.error("Failed to fetch ATT&CK data")
            return False

        # Calculate content hash for integrity verification
        content_hash = hashlib.sha256(json.dumps(attack_data).encode()).hexdigest()

        # Verify number of objects is reasonable
        if "objects" not in attack_data or not isinstance(attack_data["objects"], list):
            logger.error("Invalid ATT&CK data format: 'objects' missing or not a list")
            return False

        object_count = len(attack_data["objects"])
        if (
            object_count < 100
        ):  # Sanity check - ATT&CK data should have hundreds of objects
            logger.error(f"Suspicious object count in ATT&CK data: {object_count}")
            return False

        # Find only changed objects if we have previous data
        if "objects" in last_update_info and isinstance(
            last_update_info["objects"], dict
        ):
            # Create lookup of existing objects by ID
            existing_objects = last_update_info["objects"]

            # Find new and modified objects
            new_objects = []
            modified_objects = []
            unchanged_count = 0

            for obj in attack_data["objects"]:
                obj_id = obj.get("id")

                if not obj_id:
                    continue

                # Calculate object hash
                obj_hash = hashlib.sha256(
                    json.dumps(obj, sort_keys=True).encode()
                ).hexdigest()

                if obj_id not in existing_objects:
                    new_objects.append(obj)
                elif existing_objects[obj_id] != obj_hash:
                    modified_objects.append(obj)
                else:
                    unchanged_count += 1

            logger.info(
                f"Found {len(new_objects)} new objects, {len(modified_objects)} modified objects, and {unchanged_count} unchanged objects"
            )

            # Process only new and modified objects if any
            if new_objects or modified_objects:
                self.attack_connector.process_incremental_update(
                    new_objects, modified_objects
                )
            else:
                logger.info("No changes detected in ATT&CK data")
        else:
            # Full processing for first update or if no previous object info
            self.attack_connector.process_attack_data(attack_data)

        # Update last update info
        object_hashes = {}
        for obj in attack_data["objects"]:
            obj_id = obj.get("id")
            if obj_id:
                object_hashes[obj_id] = hashlib.sha256(
                    json.dumps(obj, sort_keys=True).encode()
                ).hexdigest()

        last_update_info = {
            "last_modified": remote_last_modified,
            "update_time": datetime.utcnow().isoformat(),
            "content_hash": content_hash,
            "object_count": object_count,
            "objects": object_hashes,
        }

        with open(last_update_file, "w") as f:
            json.dump(last_update_info, f, indent=2)

        return True

    except Exception as e:
        logger.error(f"Error updating ATT&CK framework data: {e}")
        return False


def main():
    """Main function to run the integration."""
    # Load environment variables or use defaults
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687/pipe")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

    logger.info("Starting MITRE ATT&CK Integration System")

    try:
        # Initialize integration manager
        manager = IntegrationManager(neo4j_uri, neo4j_user, neo4j_password)

        # Run initial full update
        manager.run_full_update()

        # Schedule regular updates
        manager.schedule_updates()

        # Run scheduler
        manager.run_scheduler()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error in integration system: {e}")
    finally:
        # Ensure connections are closed
        try:
            manager.close()
        except:
            pass


if __name__ == "__main__":
    main()
