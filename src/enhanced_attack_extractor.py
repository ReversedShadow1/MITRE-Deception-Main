"""
Enhanced MITRE ATT&CK Technique Extractor with Neo4j Integration
----------------------------------------------------------------
Identifies ATT&CK techniques from text using multiple extraction methods
and leverages Neo4j database for enhanced context and accuracy.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from neo4j import GraphDatabase
from prometheus_client import Counter, Histogram
from requests import Request
from starlette_prometheus import PrometheusMiddleware, metrics

from src.database.models import AnalysisResponse
from src.database.postgresql import get_db

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"attack_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logger = logging.getLogger("AtomicPipe")


class Neo4jConnector:
    """Handles connections and operations with the Neo4j database."""

    def __init__(self, uri: str, user: str, password: str, database: str = "pipe"):
        """Initialize the Neo4j connector.

        Args:
            uri: The URI for the Neo4j instance
            user: Username for the Neo4j instance
            password: Password for the Neo4j instance
            database: The Neo4j database name
        """
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
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (n) RETURN count(n) AS count LIMIT 1")
                count = result.single()["count"]
                logger.info(
                    f"Successfully connected to Neo4j database. Node count: {count}"
                )
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
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return []

    def get_attack_technique(self, technique_id: str) -> Dict:
        """Get ATT&CK technique details from the database.

        Args:
            technique_id: The ATT&CK technique ID (e.g., T1123)

        Returns:
            Dictionary with technique details
        """
        query = """
        MATCH (t:AttackTechnique {technique_id: $technique_id})
        OPTIONAL MATCH (t)-[:BELONGS_TO]->(tactic:AttackTactic)
        RETURN 
            t.technique_id as technique_id,
            t.name as name,
            t.description as description,
            t.is_subtechnique as is_subtechnique,
            t.parent_technique_id as parent_technique_id,
            t.url as url,
            collect(DISTINCT tactic.name) as tactics
        """
        params = {"technique_id": technique_id}
        results = self.run_query(query, params)
        return results[0] if results else {}

    def get_all_technique_ids(self) -> List[str]:
        """Get all ATT&CK technique IDs from the database.

        Returns:
            List of technique IDs
        """
        query = """
        MATCH (t:AttackTechnique)
        RETURN t.technique_id as technique_id
        """
        results = self.run_query(query)
        return [record["technique_id"] for record in results]

    def get_all_techniques(self) -> Dict:
        """Get all ATT&CK techniques from the database.

        Returns:
            Dictionary of technique data
        """
        query = """
        MATCH (t:AttackTechnique)
        OPTIONAL MATCH (t)-[:BELONGS_TO]->(tactic:AttackTactic)
        RETURN 
            t.technique_id as technique_id,
            t.name as name,
            t.description as description,
            t.is_subtechnique as is_subtechnique,
            t.parent_technique_id as parent_technique_id,
            t.url as url,
            collect(DISTINCT tactic.name) as tactics
        """
        results = self.run_query(query)
        techniques = {}
        for tech in results:
            tech_id = tech.get("technique_id")
            if tech_id:
                techniques[tech_id] = tech

        return techniques

    def get_technique_keywords(self, technique_id: str = None) -> Dict:
        """Get keywords for techniques from the database.

        Args:
            technique_id: Optional technique ID to filter by

        Returns:
            Dictionary mapping technique IDs to keywords
        """
        # Try to find keywords in graph properties if available
        # Otherwise return empty dict - will be loaded from file
        try:
            if technique_id:
                query = """
                MATCH (t:AttackTechnique {technique_id: $technique_id})
                RETURN 
                    t.technique_id as technique_id,
                    t.keywords as keywords
                """
                params = {"technique_id": technique_id}
                results = self.run_query(query, params)

                if results and results[0].get("keywords"):
                    keywords = {}
                    keywords[technique_id] = json.loads(results[0]["keywords"])
                    return keywords
            else:
                query = """
                MATCH (t:AttackTechnique)
                WHERE t.keywords IS NOT NULL
                RETURN 
                    t.technique_id as technique_id,
                    t.keywords as keywords
                """
                results = self.run_query(query)

                if results:
                    keywords = {}
                    for result in results:
                        tech_id = result.get("technique_id")
                        if tech_id and result.get("keywords"):
                            try:
                                keywords[tech_id] = json.loads(result["keywords"])
                            except:
                                # If not JSON, assume it's a comma-separated list
                                if isinstance(result["keywords"], str):
                                    keywords[tech_id] = [
                                        k.strip() for k in result["keywords"].split(",")
                                    ]

                    if keywords:
                        return keywords
        except Exception as e:
            logger.warning(f"Error retrieving keywords from database: {e}")
            # Continue and load from file instead

        return {}

    def get_techniques_for_cve(self, cve_id: str) -> List[Dict]:
        """Get ATT&CK techniques associated with a specific CVE.

        Args:
            cve_id: The CVE ID (e.g., CVE-2021-44228)

        Returns:
            List of dictionaries with technique details
        """
        query = """
        MATCH (cve:CVE {cve_id: $cve_id})-[r:ENABLES]->(t:AttackTechnique)
        RETURN 
            t.technique_id as technique_id,
            t.name as name,
            r.capability_description as description,
            r.mapping_type as mapping_type,
            r.source as source,
            'kev' as method,
            0.9 as confidence
        """
        params = {"cve_id": cve_id}
        return self.run_query(query, params)

    def get_techniques_for_cwe(self, cwe_id: str) -> List[Dict]:
        """Get ATT&CK techniques associated with a specific CWE.

        Args:
            cwe_id: The CWE ID (e.g., CWE-79)

        Returns:
            List of dictionaries with technique details
        """
        query = """
        MATCH (cwe:CWE {cwe_id: $cwe_id})<-[:EXHIBITS_WEAKNESS]-(cve:CVE)-[r:ENABLES]->(t:AttackTechnique)
        RETURN 
            t.technique_id as technique_id,
            t.name as name,
            count(DISTINCT cve) as cve_count,
            'cwe_derived' as method,
            0.7 as confidence
        ORDER BY cve_count DESC
        """
        params = {"cwe_id": cwe_id}
        return self.run_query(query, params)

    def get_related_techniques(
        self, technique_id: str, relationship_types: List[str] = None
    ) -> List[Dict]:
        """Get techniques related to a given technique.

        Args:
            technique_id: The ATT&CK technique ID
            relationship_types: Types of relationships to consider

        Returns:
            List of dictionaries with related technique details
        """
        if not relationship_types:
            relationship_types = ["RELATED_TO", "SUBTECHNIQUE_OF", "SIMILAR_TO"]

        relationship_query = "|".join(
            [f":{rel_type}" for rel_type in relationship_types]
        )

        query = f"""
        MATCH (t:AttackTechnique {{technique_id: $technique_id}})-[r {relationship_query}]-(related:AttackTechnique)
        RETURN 
            related.technique_id as technique_id,
            related.name as name,
            type(r) as relationship_type,
            0.6 as confidence,
            'graph_relationship' as method
        """
        params = {"technique_id": technique_id}
        return self.run_query(query, params)

    def get_capec_attack_mappings(self, text: str) -> List[Dict]:
        """Get ATT&CK techniques associated with CAPEC patterns mentioned in text.

        Args:
            text: The input text

        Returns:
            List of dictionaries with technique details
        """
        # Extract CAPEC IDs from text
        capec_pattern = r"CAPEC-(\d+)"
        capec_matches = re.findall(capec_pattern, text)

        if not capec_matches:
            return []

        capec_ids = [f"CAPEC-{capec}" for capec in capec_matches]
        query = """
        MATCH (c:CAPEC)-[:MAPPED_TO|SIMILAR_TO]->(t:AttackTechnique)
        WHERE c.capec_id IN $capec_ids
        RETURN 
            t.technique_id as technique_id,
            t.name as name,
            c.capec_id as capec_id,
            c.name as capec_name,
            'capec_mapping' as method,
            0.8 as confidence
        """
        params = {"capec_ids": capec_ids}
        return self.run_query(query, params)

    def enrich_techniques(self, techniques: List[Dict]) -> List[Dict]:
        """Enrich techniques with additional information from the database.

        Args:
            techniques: List of dictionaries with technique details

        Returns:
            Enriched list of dictionaries
        """
        # Skip if empty
        if not techniques:
            return []

        # Get unique technique IDs
        technique_ids = [t["technique_id"] for t in techniques]

        # Query for full technique details
        query = """
        MATCH (t:AttackTechnique)
        WHERE t.technique_id IN $technique_ids
        OPTIONAL MATCH (t)-[:BELONGS_TO]->(tactic:AttackTactic)
        OPTIONAL MATCH (m:AttackMitigation)-[:MITIGATES]->(t)
        RETURN 
            t.technique_id as technique_id,
            t.name as name,
            t.description as description,
            t.url as url,
            collect(DISTINCT tactic.name) as tactics,
            collect(DISTINCT tactic.tactic_id) as tactic_ids,
            collect(DISTINCT m.mitigation_id + ': ' + m.name) as mitigations
        """
        params = {"technique_ids": technique_ids}
        enriched_data = self.run_query(query, params)

        # Create lookup dictionary
        enriched_lookup = {t["technique_id"]: t for t in enriched_data}

        # Enrich original techniques
        for technique in techniques:
            tech_id = technique["technique_id"]
            if tech_id in enriched_lookup:
                enriched = enriched_lookup[tech_id]
                # Add enriched data while preserving original fields
                for key, value in enriched.items():
                    if key not in technique or not technique[key]:
                        technique[key] = value

        return techniques


from src.data_manager import ATTCKDataLoader
from src.extractors.bm25_extractor import BM25Extractor
from src.extractors.classifier import SecureBERTClassifier
from src.extractors.kev_extractor import KEVExtractor
from src.extractors.ner_extractor import SecureBERTNERExtractor

# Import extractor classes from original implementation
# These are imported here to ensure compatibility with existing code
from src.extractors.rule_based import RuleBasedExtractor
from src.extractors.semantic import BGEEmbeddingExtractor


class EnhancedATTCKExtractor:
    """
    Enhanced MITRE ATT&CK technique extractor with Neo4j integration
    """

    def __init__(
        self,
        data_dir: str = "data",
        models_dir: str = "models",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "pipe",
        use_gpu: bool = True,
        auto_load: bool = False,
        memory_efficient: bool = False,
    ):
        """
        Initialize the enhanced ATT&CK technique extractor

        Args:
            data_dir: Directory containing ATT&CK data files
            models_dir: Directory for model storage
            neo4j_uri: URI for the Neo4j database
            neo4j_user: Username for the Neo4j database
            neo4j_password: Password for the Neo4j database
            neo4j_database: Neo4j database name
            use_gpu: Whether to use GPU acceleration if available
            auto_load: Whether to auto-load all extractors on initialization
            memory_efficient: Whether to operate in memory-efficient mode (load/unload models)
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.memory_efficient = memory_efficient

        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # Initialize Neo4j connector
        logger.info("Initializing Neo4j connector...")
        self.neo4j = Neo4jConnector(
            neo4j_uri, neo4j_user, neo4j_password, neo4j_database
        )
        self.use_neo4j = self.neo4j.connect()

        if not self.use_neo4j:
            logger.warning("Failed to connect to Neo4j, falling back to local data")
            # Initialize the data loader with auto-loading since we can't use Neo4j
            logger.info("Initializing ATT&CK data loader...")
            self.data_loader = ATTCKDataLoader(data_dir=data_dir, auto_load=True)
            self.techniques = self.data_loader.techniques
            self.technique_keywords = self.data_loader.technique_keywords
        else:
            # Load technique data and keywords from Neo4j
            logger.info("Loading technique data from Neo4j...")
            self.techniques = self.neo4j.get_all_techniques()

            # Try to get keywords from Neo4j first, then fall back to file
            self.technique_keywords = self.neo4j.get_technique_keywords()

            # If no keywords in Neo4j, load from file
            if not self.technique_keywords:
                # Try to load from file
                keywords_file = os.path.join(data_dir, "technique_keywords.json")
                if os.path.exists(keywords_file):
                    try:
                        with open(keywords_file, "r", encoding="utf-8") as f:
                            self.technique_keywords = json.load(f)
                        logger.info(
                            f"Loaded {len(self.technique_keywords)} technique keyword mappings from file"
                        )
                    except Exception as e:
                        logger.error(f"Error loading technique keywords from file: {e}")
                        self.technique_keywords = {}

            # If still no keywords, generate basic ones
            if not self.technique_keywords:
                self.technique_keywords = self._generate_basic_keywords(self.techniques)

                # Save generated keywords
                try:
                    keywords_file = os.path.join(data_dir, "technique_keywords.json")
                    with open(keywords_file, "w", encoding="utf-8") as f:
                        json.dump(self.technique_keywords, f, indent=2)
                    logger.info(f"Saved generated keywords to {keywords_file}")
                except Exception as e:
                    logger.error(f"Error saving generated keywords: {e}")

        if not self.techniques:
            logger.error(
                "No technique data loaded! Check Neo4j connection and data directory."
            )
            raise ValueError("No technique data loaded")

        logger.info(
            f"Loaded {len(self.techniques)} ATT&CK techniques and {len(self.technique_keywords)} keyword mappings"
        )

        # Initialize extractors dictionary
        self.extractors = {}

        # Initialize with auto_load if requested
        if auto_load:
            self._load_all_extractors()
        else:
            # Always initialize rule-based and BM25 extractors (lightweight)
            self._load_rule_based()
            self._load_bm25()
            self._load_kev_extractor()

            # Other extractors will be loaded on demand
            self.extractors["ner"] = None
            self.extractors["semantic"] = None
            self.extractors["classifier"] = None

        logger.info("Enhanced ATT&CK technique extractor initialized successfully")

    def _generate_basic_keywords(self, techniques: Dict) -> Dict:
        """
        Generate basic keywords for techniques from their names and descriptions

        Args:
            techniques: Dictionary of technique data

        Returns:
            Dictionary mapping technique IDs to keywords
        """
        # Stop words to filter out
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
            "their",
            "they",
            "have",
            "has",
            "been",
            "were",
            "when",
            "where",
            "will",
            "what",
            "who",
            "how",
            "some",
            "such",
            "than",
            "then",
            "these",
            "those",
            "there",
            "about",
        }

        keywords = {}

        for tech_id, tech in techniques.items():
            # Get technique name and description
            name = tech.get("name", "")
            desc = tech.get("description", "")

            # Extract first sentence of description (usually the most relevant)
            first_sentence = desc.split(".")[0] if desc else ""

            # Extract keywords from name (lower case, split on spaces and punctuation)
            name_words = re.findall(r"\b[a-zA-Z0-9]+\b", name.lower())
            name_words = [w for w in name_words if len(w) > 3 and w not in stop_words]

            # Extract keywords from first sentence
            desc_words = re.findall(r"\b[a-zA-Z0-9]+\b", first_sentence.lower())
            desc_words = [w for w in desc_words if len(w) > 3 and w not in stop_words]

            # Include full technique name
            tech_keywords = [name.lower()]

            # Include multi-word phrases from name
            name_phrases = []
            name_parts = re.split(r"[/\(\)\[\]{}:]", name)
            for part in name_parts:
                part = part.strip()
                if part and " " in part:
                    name_phrases.append(part.lower())

            # Add relevant words from name and description
            tech_keywords.extend(name_phrases)
            tech_keywords.extend(name_words)
            tech_keywords.extend(desc_words)

            # Remove duplicates and keep only unique keywords
            tech_keywords = list(set(tech_keywords))

            # Store keywords for this technique
            keywords[tech_id] = tech_keywords

        logger.info(f"Generated {len(keywords)} basic keyword mappings")
        return keywords

    def _load_all_extractors(self):
        """Load all available extractors"""
        logger.info("Loading all extractors...")
        self._load_rule_based()
        self._load_bm25()
        self._load_ner()
        self._load_semantic()
        self._load_classifier()
        self._load_kev_extractor()

    def _load_rule_based(self):
        """Load the rule-based extractor"""
        logger.info("Loading rule-based extractor...")
        self.extractors["rule_based"] = RuleBasedExtractor(
            technique_keywords=self.technique_keywords,
            techniques_data=self.techniques,
            neo4j_connector=self.neo4j if self.use_neo4j else None,
        )
        return self.extractors["rule_based"]

    def _load_bm25(self):
        """Load the BM25 extractor"""
        logger.info("Loading BM25 extractor...")
        bm25_dir = os.path.join(self.models_dir, "bm25")
        os.makedirs(bm25_dir, exist_ok=True)

        self.extractors["bm25"] = BM25Extractor(
            techniques=self.techniques,
            technique_keywords=self.technique_keywords,
            models_dir=bm25_dir,
            bm25_variant="plus",  # Use the enhanced BM25Plus variant
            neo4j_connector=self.neo4j if self.use_neo4j else None,
        )
        return self.extractors["bm25"]

    def _load_ner(self):
        """Load the NER extractor"""
        if self.extractors.get("ner") is None:
            try:
                logger.info("Loading NER extractor...")
                start_time = time.time()
                # Create cache directory
                ner_cache_dir = os.path.join(self.models_dir, "SecureBERT-NER")
                os.makedirs(ner_cache_dir, exist_ok=True)

                self.extractors["ner"] = SecureBERTNERExtractor(
                    model_name="CyberPeace-Institute/SecureBERT-NER",  # Direct from Hugging Face
                    techniques_data=self.techniques,
                    technique_keywords=self.technique_keywords,
                    cache_dir=ner_cache_dir,
                    use_gpu=self.use_gpu,
                    neo4j_connector=self.neo4j if self.use_neo4j else None,
                )

                # Pre-load the model
                if not self.memory_efficient:
                    logger.info("Pre-loading NER model...")
                    success = self.extractors["ner"].load_model()
                    if not success:
                        logger.warning("Failed to load NER model")
                        self.extractors["ner"] = None
                        return None

                # Track model load time for monitoring
                load_time = time.time() - start_time
                try:
                    MODEL_LOAD_TIME.labels(model_type="ner").observe(load_time)
                except Exception as e:
                    logger.warning(f"Error recording model load time metric: {e}")

                logger.info(f"NER extractor loaded in {load_time:.2f}s")

            except Exception as e:
                logger.warning(f"Error loading NER extractor: {str(e)}")
                self.extractors["ner"] = None
                return None

        return self.extractors["ner"]

    def _load_semantic(self):
        """Load the semantic search extractor"""
        if self.extractors.get("semantic") is None:
            try:
                logger.info("Loading semantic search extractor...")
                start_time = time.time()
                # Create model directory
                bge_dir = os.path.join(self.models_dir, "bge-large-en-v1.5")
                os.makedirs(bge_dir, exist_ok=True)

                self.extractors["semantic"] = BGEEmbeddingExtractor(
                    model_name="BAAI/bge-large-en-v1.5",  # Direct from Hugging Face
                    techniques=self.techniques,
                    technique_keywords=self.technique_keywords,
                    models_dir=bge_dir,
                    use_gpu=self.use_gpu,
                    neo4j_connector=self.neo4j if self.use_neo4j else None,
                )

                # Pre-load the model if not in memory-efficient mode
                if not self.memory_efficient:
                    logger.info("Pre-loading semantic model...")
                    success = self.extractors["semantic"].load_model()
                    if not success:
                        logger.warning("Failed to load semantic model")
                        self.extractors["semantic"] = None
                        return None

                # Track model load time for monitoring
                load_time = time.time() - start_time
                try:
                    MODEL_LOAD_TIME.labels(model_type="semantic").observe(load_time)
                except Exception as e:
                    logger.warning(f"Error recording model load time metric: {e}")

                logger.info(f"Semantic extractor loaded in {load_time:.2f}s")

            except Exception as e:
                logger.warning(f"Error loading semantic extractor: {str(e)}")
                self.extractors["semantic"] = None
                return None

        return self.extractors["semantic"]

    def _load_classifier(self):
        """Load the classifier model"""
        if self.extractors.get("classifier") is None:
            try:
                logger.info("Loading classifier...")
                start_time = time.time()
                classifier_dir = os.path.join(self.models_dir, "secureBERT")
                embeddings_dir = os.path.join(self.models_dir, "embeddings")
                os.makedirs(classifier_dir, exist_ok=True)
                os.makedirs(embeddings_dir, exist_ok=True)

                self.extractors["classifier"] = SecureBERTClassifier(
                    techniques_data=self.techniques,
                    cache_dir=classifier_dir,
                    embeddings_dir=embeddings_dir,
                    use_gpu=self.use_gpu,
                    neo4j_connector=self.neo4j if self.use_neo4j else None,
                )

                # Load the base model if not in memory-efficient mode
                if not self.memory_efficient:
                    logger.info("Loading base model for classifier...")
                    success = self.extractors["classifier"].load_base_model()
                    if not success:
                        logger.error("Failed to load classifier base model")
                        self.extractors["classifier"] = None
                        return None

                # Track model load time for monitoring
                load_time = time.time() - start_time
                try:
                    MODEL_LOAD_TIME.labels(model_type="classifier").observe(load_time)
                except Exception as e:
                    logger.warning(f"Error recording model load time metric: {e}")

                logger.info(f"Classifier loaded in {load_time:.2f}s")

            except Exception as e:
                logger.error(f"Error loading classifier: {str(e)}", exc_info=True)
                self.extractors["classifier"] = None
                return None

        return self.extractors["classifier"]

    def _load_kev_extractor(self):
        """Load the KEV-based extractor"""
        logger.info("Loading KEV extractor...")

        try:
            from src.integrations.kev_mapper import KEVMapper

            kev_mapper = KEVMapper(data_dir=self.data_dir)
            kev_mapper.load_kev_data()
            kev_mapper.load_cve_attack_mappings()
            # Removed: kev_mapper.generate_cwe_technique_mappings(self.techniques)

            self.extractors["kev"] = KEVExtractor(
                kev_mapper=kev_mapper, techniques_data=self.techniques
            )

            logger.info("KEV extractor loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load KEV extractor: {e}")
            logger.info("Continuing without KEV extractor")
            self.extractors["kev"] = None

        return self.extractors["kev"]

    def _unload_memory_intensive_models(self):
        """Unload memory-intensive models to free resources"""
        if self.memory_efficient:
            logger.info("Unloading memory-intensive models...")

            # Unload NER model if loaded
            if self.extractors.get("ner") is not None:
                self.extractors["ner"].unload_model()

            # Unload semantic model if loaded
            if self.extractors.get("semantic") is not None:
                self.extractors["semantic"].unload_model()

            # Unload classifier models if loaded
            if self.extractors.get("classifier") is not None:
                self.extractors["classifier"].unload_models()

            # Force garbage collection
            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Memory-intensive models unloaded")

    def extract_techniques(
        self,
        text: str,
        extractors: List[str] = None,
        threshold: float = 0.2,
        top_k: int = 10,
        use_ensemble: bool = True,
        validate_results: bool = True,
    ) -> Dict:
        """
        Extract ATT&CK techniques from input text using selected extractors

        Args:
            text: The input text to analyze
            extractors: List of extractors to use (rule_based, bm25, ner, semantic, kev, classifier)
                    If None, use rule_based, bm25 and kev by default
            threshold: Minimum confidence threshold for results
            top_k: Maximum number of techniques to return
            use_ensemble: Whether to use ensemble method for combining results

        Returns:
            Dictionary with extraction results
        """
        # Default extractors if none specified
        if extractors is None:
            extractors = ["rule_based", "bm25", "kev"]

        start_time = time.time()

        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided")
            return {
                "techniques": [],
                "meta": {
                    "text_length": 0,
                    "processing_time": 0,
                    "extractors_used": {},
                    "error": "Empty text provided",
                },
            }

        logger.info(
            f"Extracting techniques from text of length {len(text)} using extractors: {', '.join(extractors)}"
        )

        try:
            # Dictionary to store results from individual extractors
            extractor_results = {}

            # Get results from each requested extractor
            for extractor_name in extractors:
                if (
                    extractor_name not in self.extractors
                    or self.extractors[extractor_name] is None
                ):
                    # Load on demand
                    if extractor_name == "rule_based":
                        self._load_rule_based()
                    elif extractor_name == "bm25":
                        self._load_bm25()
                    elif extractor_name == "ner":
                        self._load_ner()
                    elif extractor_name == "semantic":
                        self._load_semantic()
                    elif extractor_name == "classifier":
                        self._load_classifier()
                    elif extractor_name == "kev":
                        self._load_kev_extractor()
                    else:
                        logger.warning(f"Unknown extractor: {extractor_name}")
                        continue

                # Get extractor
                extractor = self.extractors[extractor_name]

                if extractor is None:
                    logger.warning(f"Extractor {extractor_name} not properly loaded")
                    continue

                # Extract techniques using the current extractor
                logger.info(f"Extracting techniques using {extractor_name}...")

                try:
                    # Handle different method signatures based on extractor type
                    if extractor_name == "rule_based" or extractor_name == "ner":
                        # These extractors expect min_confidence and max_results
                        results = extractor.extract_techniques(
                            text=text, min_confidence=threshold, max_results=top_k * 2
                        )
                    else:
                        # Other extractors expect threshold and top_k
                        results = extractor.extract_techniques(
                            text=text, threshold=threshold, top_k=top_k * 2
                        )

                    # Store results
                    extractor_results[extractor_name] = results
                    logger.info(
                        f"{extractor_name} extractor found {len(results)} techniques"
                    )

                except Exception as e:
                    logger.error(
                        f"Error extracting techniques with {extractor_name}: {str(e)}",
                        exc_info=True,
                    )
                    extractor_results[extractor_name] = []

            # Use ensemble method if requested
            final_results = []

            if (
                use_ensemble
                and "classifier" in extractors
                and self.extractors["classifier"] is not None
            ):
                logger.info("Using ensemble method to combine results...")
                classifier = self.extractors["classifier"]

                # Get results from each extractor
                rule_results = extractor_results.get("rule_based", [])
                bm25_results = extractor_results.get("bm25", [])
                ner_results = extractor_results.get("ner", [])
                semantic_results = extractor_results.get("semantic", [])
                kev_results = extractor_results.get("kev", [])

                # Use ensemble method from classifier
                # Check if we're using Neo4j-backed enhanced ensemble
                if hasattr(classifier, "ensemble_extractors") and callable(
                    getattr(classifier, "ensemble_extractors")
                ):
                    # Use extended ensemble method if available
                    ensemble_method = getattr(classifier, "ensemble_extractors")
                    final_results = ensemble_method(
                        text=text,
                        rule_results=(
                            rule_results if "rule_based" in extractors else None
                        ),
                        bm25_results=bm25_results if "bm25" in extractors else None,
                        ner_results=ner_results if "ner" in extractors else None,
                        semantic_results=(
                            semantic_results if "semantic" in extractors else None
                        ),
                        kev_results=kev_results if "kev" in extractors else None,
                        threshold=threshold,
                        max_results=top_k,
                    )
                else:
                    # Fall back to original ensemble method
                    ensemble_method = getattr(classifier, "ensemble_extractors")
                    final_results = ensemble_method(
                        text=text,
                        rule_results=(
                            rule_results if "rule_based" in extractors else None
                        ),
                        bm25_results=bm25_results if "bm25" in extractors else None,
                        ner_results=ner_results if "ner" in extractors else None,
                        threshold=threshold,
                        max_results=top_k,
                    )

                    # Add KEV results separately if they weren't included in the ensemble
                    if kev_results:
                        # Add any KEV results not already in final_results
                        final_tech_ids = [r["technique_id"] for r in final_results]
                        for kev_result in kev_results:
                            if kev_result["technique_id"] not in final_tech_ids:
                                final_results.append(kev_result)

                        # Re-sort by confidence
                        final_results.sort(key=lambda x: x["confidence"], reverse=True)
                        final_results = final_results[:top_k]
            else:
                # Combine results from all extractors manually
                logger.info("Combining results from all extractors...")
                all_results = []
                for results in extractor_results.values():
                    all_results.extend(results)

                final_results = self._combine_results(all_results, threshold)

                # Sort by confidence and limit to top_k
                final_results = sorted(
                    final_results, key=lambda x: x["confidence"], reverse=True
                )[:top_k]

            # Enrich technique details if using Neo4j
            if self.use_neo4j:
                final_results = self.neo4j.enrich_techniques(final_results)
            else:
                # Add technique details from local data
                for result in final_results:
                    tech_id = result["technique_id"]
                    if tech_id in self.techniques:
                        tech_data = self.techniques[tech_id]
                        for key, value in tech_data.items():
                            if key not in result or not result[key]:
                                result[key] = value

            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time

            # Build the result object
            result_object = {
                "techniques": final_results,
                "meta": {
                    "text_length": len(text),
                    "processing_time": round(processing_time, 3),
                    "extractors_used": {extractor: True for extractor in extractors},
                    "ensemble_used": use_ensemble and "classifier" in extractors,
                    "threshold": threshold,
                    "technique_count": len(final_results),
                    "using_neo4j": self.use_neo4j,
                },
            }

            if validate_results:
                try:
                    # Import and use the validator
                    from src.validation.quality_checks import DataQualityValidator

                    validator = DataQualityValidator(
                        techniques_data=self.techniques,
                        technique_keywords=self.technique_keywords,
                    )

                    # Validate each result
                    validation_issues = validator.validate_extraction_results(
                        final_results
                    )

                    # Check for consistency issues
                    consistency_issues = validator.validate_consistency(final_results)

                    # Add validation results to the response
                    if validation_issues or consistency_issues:
                        result_object["meta"]["validation_issues"] = {
                            "result_issues": validation_issues,
                            "consistency_issues": consistency_issues,
                        }

                        # Log validation issues
                        logger.warning(
                            f"Validation issues detected in extraction results: "
                            f"{len(validation_issues)} result issues, "
                            f"{len(consistency_issues)} consistency issues"
                        )
                except Exception as e:
                    # Don't fail the whole request if validation has issues
                    logger.error(f"Error during result validation: {str(e)}")
                    result_object["meta"]["validation_error"] = str(e)

            # Unload memory-intensive models if in memory-efficient mode
            if self.memory_efficient:
                self._unload_memory_intensive_models()

            return result_object

        except Exception as e:
            logger.error(f"Error extracting techniques: {str(e)}", exc_info=True)
            return {
                "techniques": [],
                "meta": {
                    "text_length": len(text),
                    "processing_time": time.time() - start_time,
                    "extractors_used": {extractor: True for extractor in extractors},
                    "error": str(e),
                },
            }

    # Update the extract_techniques_with_details method in EnhancedATTCKExtractor class
    # This modified version fixes the parameter inconsistencies between extractors

    def extract_techniques_with_details(
        self,
        text: str,
        extractors: List[str] = None,
        threshold: float = 0.2,
        top_k: int = 10,
        use_ensemble: bool = True,
        validate_results: bool = True,
    ) -> Dict:
        """
        Extract ATT&CK techniques from input text with detailed steps and intermediate results

        Args:
            text: The input text to analyze
            extractors: List of extractors to use
            threshold: Minimum confidence threshold for results
            top_k: Maximum number of techniques to return
            use_ensemble: Whether to use ensemble method for combining results
            validate_results: Whether to validate results

        Returns:
            Dictionary with extraction results including all intermediate steps
        """
        # Default extractors if none specified
        if extractors is None:
            extractors = ["rule_based", "bm25", "kev"]

        start_time = time.time()

        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided")
            return {
                "techniques": [],
                "meta": {
                    "text_length": 0,
                    "processing_time": 0,
                    "extractors_used": {},
                    "error": "Empty text provided",
                },
            }

        logger.info(
            f"Extracting techniques from text of length {len(text)} using extractors: {', '.join(extractors)}"
        )

        try:
            # Dictionary to store results from individual extractors
            extractor_results = {}
            extraction_steps = []

            # Get results from each requested extractor
            for extractor_name in extractors:
                extractor_start_time = time.time()

                extraction_steps.append(
                    {
                        "step": f"Starting {extractor_name} extraction",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                if (
                    extractor_name not in self.extractors
                    or self.extractors[extractor_name] is None
                ):
                    # Load on demand
                    if extractor_name == "rule_based":
                        self._load_rule_based()
                    elif extractor_name == "bm25":
                        self._load_bm25()
                    elif extractor_name == "ner":
                        self._load_ner()
                    elif extractor_name == "semantic":
                        self._load_semantic()
                    elif extractor_name == "classifier":
                        self._load_classifier()
                    elif extractor_name == "kev":
                        self._load_kev_extractor()
                    else:
                        logger.warning(f"Unknown extractor: {extractor_name}")
                        extraction_steps.append(
                            {
                                "step": f"Failed to load unknown extractor: {extractor_name}",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        continue

                    extraction_steps.append(
                        {
                            "step": f"Loaded {extractor_name} extractor",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                # Get extractor
                extractor = self.extractors[extractor_name]

                if extractor is None:
                    logger.warning(f"Extractor {extractor_name} not properly loaded")
                    extraction_steps.append(
                        {
                            "step": f"Extractor {extractor_name} not properly loaded",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    continue

                # Extract techniques using the current extractor
                logger.info(f"Extracting techniques using {extractor_name}...")
                extraction_steps.append(
                    {
                        "step": f"Starting extraction with {extractor_name}",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                try:
                    # Handle different parameter naming conventions for each extractor
                    # This is the key fix - each extractor type gets the right parameters
                    if (
                        extractor_name == "rule_based"
                        or extractor_name == "ner"
                        or extractor_name == "kev"
                    ):
                        # These extractors expect min_confidence instead of threshold
                        results = extractor.extract_techniques(
                            text=text, min_confidence=threshold, max_results=top_k
                        )
                    elif extractor_name == "bm25" or extractor_name == "semantic":
                        # These extractors expect threshold instead of min_confidence
                        results = extractor.extract_techniques(
                            text=text, threshold=threshold, top_k=top_k
                        )
                    elif extractor_name == "classifier":
                        # Special case for classifier - may not need extraction parameters
                        if hasattr(extractor, "extract_techniques"):
                            results = extractor.extract_techniques(
                                text=text, threshold=threshold, top_k=top_k
                            )
                        else:
                            # If it doesn't have an extract_techniques method, use a simple pass-through
                            results = []
                    else:
                        # Fallback to generic parameters - may not work for all extractors
                        results = extractor.extract_techniques(text=text)

                    # Store results
                    extractor_results[extractor_name] = results

                    extraction_steps.append(
                        {
                            "step": f"{extractor_name} found {len(results)} techniques",
                            "timestamp": datetime.now().isoformat(),
                            "duration_ms": round(
                                (time.time() - extractor_start_time) * 1000, 2
                            ),
                            "technique_count": len(results),
                        }
                    )

                    logger.info(
                        f"{extractor_name} extractor found {len(results)} techniques"
                    )

                except Exception as e:
                    logger.error(
                        f"Error extracting techniques with {extractor_name}: {str(e)}",
                        exc_info=True,
                    )
                    extractor_results[extractor_name] = []
                    extraction_steps.append(
                        {
                            "step": f"Error in {extractor_name} extraction: {str(e)}",
                            "timestamp": datetime.now().isoformat(),
                            "error": str(e),
                        }
                    )

            # Use ensemble method if requested
            final_results = []
            combination_start_time = time.time()

            extraction_steps.append(
                {
                    "step": "Starting results combination phase",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if (
                use_ensemble
                and "classifier" in extractors
                and self.extractors["classifier"] is not None
            ):
                logger.info("Using ensemble method to combine results...")
                extraction_steps.append(
                    {
                        "step": "Using classifier ensemble method",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                classifier = self.extractors["classifier"]

                # Get results from each extractor
                rule_results = extractor_results.get("rule_based", [])
                bm25_results = extractor_results.get("bm25", [])
                ner_results = extractor_results.get("ner", [])
                kev_results = extractor_results.get("kev", [])

                # Check what parameters the ensemble_extractors method accepts
                import inspect

                if hasattr(classifier, "ensemble_extractors"):
                    sig = inspect.signature(classifier.ensemble_extractors)
                    params = sig.parameters.keys()

                    extraction_steps.append(
                        {
                            "step": f"Ensemble method accepts parameters: {', '.join(params)}",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                    # Build appropriate parameters based on what the method accepts
                    ensemble_params = {
                        "text": text,
                        "threshold": threshold,
                        "max_results": top_k,
                    }

                    if "rule_results" in params:
                        ensemble_params["rule_results"] = (
                            rule_results if "rule_based" in extractors else None
                        )

                    if "bm25_results" in params:
                        ensemble_params["bm25_results"] = (
                            bm25_results if "bm25" in extractors else None
                        )

                    if "ner_results" in params:
                        ensemble_params["ner_results"] = (
                            ner_results if "ner" in extractors else None
                        )

                    if "semantic_results" in params:
                        ensemble_params["semantic_results"] = (
                            extractor_results.get("semantic", [])
                            if "semantic" in extractors
                            else None
                        )

                    if "kev_results" in params:
                        ensemble_params["kev_results"] = (
                            kev_results if "kev" in extractors else None
                        )

                    # Call ensemble method with only the parameters it accepts
                    final_results = classifier.ensemble_extractors(**ensemble_params)

                    extraction_steps.append(
                        {
                            "step": "Applied ensemble method with compatible parameters",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                else:
                    extraction_steps.append(
                        {
                            "step": "Classifier does not have ensemble_extractors method, using manual combination",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    # Fall back to manual combination
                    all_results = []
                    for results in extractor_results.values():
                        all_results.extend(results)
                    final_results = self._combine_results(all_results, threshold)
            else:
                # Combine results from all extractors manually
                logger.info("Combining results from all extractors...")
                extraction_steps.append(
                    {
                        "step": "Manually combining results from all extractors",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                all_results = []
                for extractor_name, results in extractor_results.items():
                    all_results.extend(results)

                final_results = self._combine_results(all_results, threshold)
                extraction_steps.append(
                    {
                        "step": f"Combined {len(all_results)} results into {len(final_results)} unique techniques",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Sort by confidence and limit to top_k
            final_results = sorted(
                final_results, key=lambda x: x.get("confidence", 0), reverse=True
            )[:top_k]

            extraction_steps.append(
                {
                    "step": "Finished combining results",
                    "timestamp": datetime.now().isoformat(),
                    "duration_ms": round(
                        (time.time() - combination_start_time) * 1000, 2
                    ),
                    "final_technique_count": len(final_results),
                }
            )

            # Enrich technique details if using Neo4j
            enrichment_start_time = time.time()
            extraction_steps.append(
                {
                    "step": "Starting technique enrichment",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if self.use_neo4j:
                final_results = self.neo4j.enrich_techniques(final_results)
                extraction_steps.append(
                    {
                        "step": "Enriched techniques using Neo4j",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            else:
                # Add technique details from local data
                for result in final_results:
                    tech_id = result.get("technique_id")
                    if tech_id in self.techniques:
                        tech_data = self.techniques[tech_id]
                        for key, value in tech_data.items():
                            if key not in result or not result[key]:
                                result[key] = value
                extraction_steps.append(
                    {
                        "step": "Enriched techniques using local data",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            extraction_steps.append(
                {
                    "step": "Completed technique enrichment",
                    "timestamp": datetime.now().isoformat(),
                    "duration_ms": round(
                        (time.time() - enrichment_start_time) * 1000, 2
                    ),
                }
            )

            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time

            # Build the result object
            result_object = {
                "techniques": final_results,
                "individual_extractor_results": extractor_results,
                "extraction_steps": extraction_steps,
                "meta": {
                    "text_length": len(text),
                    "processing_time": round(processing_time, 3),
                    "extractors_used": {extractor: True for extractor in extractors},
                    "ensemble_used": use_ensemble and "classifier" in extractors,
                    "threshold": threshold,
                    "technique_count": len(final_results),
                    "using_neo4j": self.use_neo4j,
                },
            }

            # Validate results if requested
            if validate_results:
                validation_start_time = time.time()
                extraction_steps.append(
                    {
                        "step": "Starting result validation",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                try:
                    # Import and use the validator
                    from src.validation.quality_checks import DataQualityValidator

                    validator = DataQualityValidator(
                        techniques_data=self.techniques,
                        technique_keywords=self.technique_keywords,
                    )

                    # Validate each result
                    validation_issues = validator.validate_extraction_results(
                        final_results
                    )

                    # Check for consistency issues
                    consistency_issues = validator.validate_consistency(final_results)

                    # Add validation results to the response
                    if validation_issues or consistency_issues:
                        result_object["meta"]["validation_issues"] = {
                            "result_issues": validation_issues,
                            "consistency_issues": consistency_issues,
                        }

                        extraction_steps.append(
                            {
                                "step": f"Validation found {len(validation_issues)} result issues and {len(consistency_issues)} consistency issues",
                                "timestamp": datetime.now().isoformat(),
                                "duration_ms": round(
                                    (time.time() - validation_start_time) * 1000, 2
                                ),
                            }
                        )

                        # Log validation issues
                        logger.warning(
                            f"Validation issues detected in extraction results: "
                            f"{len(validation_issues)} result issues, "
                            f"{len(consistency_issues)} consistency issues"
                        )
                    else:
                        extraction_steps.append(
                            {
                                "step": "Validation completed with no issues",
                                "timestamp": datetime.now().isoformat(),
                                "duration_ms": round(
                                    (time.time() - validation_start_time) * 1000, 2
                                ),
                            }
                        )
                except Exception as e:
                    # Don't fail the whole request if validation has issues
                    logger.error(f"Error during result validation: {str(e)}")
                    result_object["meta"]["validation_error"] = str(e)
                    extraction_steps.append(
                        {
                            "step": f"Validation error: {str(e)}",
                            "timestamp": datetime.now().isoformat(),
                            "error": str(e),
                        }
                    )

            # Unload memory-intensive models if in memory-efficient mode
            if self.memory_efficient:
                extraction_steps.append(
                    {
                        "step": "Unloading memory-intensive models",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                self._unload_memory_intensive_models()

            # Add final step
            extraction_steps.append(
                {
                    "step": "Extraction process completed",
                    "timestamp": datetime.now().isoformat(),
                    "total_duration_ms": round(processing_time * 1000, 2),
                }
            )

            # Update extraction steps in result
            result_object["extraction_steps"] = extraction_steps

            return result_object

        except Exception as e:
            logger.error(f"Error extracting techniques: {str(e)}", exc_info=True)
            return {
                "techniques": [],
                "individual_extractor_results": {},
                "extraction_steps": [
                    {
                        "step": "Error in extraction process",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                    }
                ],
                "meta": {
                    "text_length": len(text),
                    "processing_time": time.time() - start_time,
                    "extractors_used": {extractor: True for extractor in extractors},
                    "error": str(e),
                },
            }

    def _combine_results(
        self, results: List[Dict], threshold: float = 0.0
    ) -> List[Dict]:
        """
        Combine results from multiple extractors with intelligent confidence merging

        Args:
            results: List of technique results from various extractors
            threshold: Minimum confidence threshold

        Returns:
            Combined and deduplicated results list
        """
        if not results:
            return []

        combined = {}

        # Get set of valid technique IDs
        valid_technique_ids = set(self.techniques.keys())

        # Method weights based on empirical performance
        method_weights = {
            "rule_based": 0.7,
            "bm25": 0.8,
            "ner": 0.9,
            "semantic": 0.85,
            "classifier": 0.8,
            "kev": 0.95,  # KEV is highly reliable when available
        }

        # Context-specific boosting factors
        for result in results:
            tech_id = result["technique_id"]

            # Skip invalid technique IDs
            if tech_id not in valid_technique_ids:
                logger.warning(f"Skipping invalid technique ID: {tech_id}")
                continue

            confidence = result.get("confidence", 0.5)  # Default if missing
            method = result.get("method", "unknown")

            # Apply method-specific weight
            weight = method_weights.get(
                method, 0.6
            )  # Default weight for unknown methods
            weighted_confidence = confidence * weight

            if weighted_confidence < threshold:
                continue

            # Apply contextual boosting
            contextual_boost = 1.0

            # Boost CVE-based detections (likely more reliable)
            if "cve_id" in result:
                contextual_boost *= 1.15

            # Boost if technique has strong textual evidence
            if (
                "matched_keywords" in result
                and len(result.get("matched_keywords", [])) > 3
            ):
                contextual_boost *= 1.1

            # Apply the boost to confidence
            boosted_confidence = min(weighted_confidence * contextual_boost, 1.0)

            if tech_id not in combined:
                # First occurrence of this technique
                combined[tech_id] = {
                    "technique_id": tech_id,
                    "confidence": boosted_confidence,
                    "method": method,
                    "methods": [method],
                    "scores": [confidence],
                    "weighted_scores": [weighted_confidence],
                    "count": 1,
                }

                # Add CVE information if present
                if "cve_id" in result:
                    combined[tech_id]["cve_id"] = result["cve_id"]

            else:
                # Update existing technique
                combined[tech_id]["count"] += 1

                # Add method if not already present
                if method not in combined[tech_id]["methods"]:
                    combined[tech_id]["methods"].append(method)
                    combined[tech_id]["method"] = "+".join(
                        sorted(set(combined[tech_id]["methods"]))
                    )

                # Add confidence score
                combined[tech_id]["scores"].append(confidence)
                combined[tech_id]["weighted_scores"].append(weighted_confidence)

                # Add CVE information if present
                if "cve_id" in result and "cve_id" not in combined[tech_id]:
                    combined[tech_id]["cve_id"] = result["cve_id"]
                elif "cve_id" in result and "cve_id" in combined[tech_id]:
                    # Create a list of CVEs if multiple
                    if "cve_ids" not in combined[tech_id]:
                        combined[tech_id]["cve_ids"] = [combined[tech_id]["cve_id"]]

                    # Add new CVE if not already in list
                    if result["cve_id"] not in combined[tech_id]["cve_ids"]:
                        combined[tech_id]["cve_ids"].append(result["cve_id"])

                # Update confidence - weighted average with boost for multiple extractors
                num_methods = len(combined[tech_id]["methods"])
                if num_methods > 1:
                    # Boost confidence when multiple methods agree
                    # Use a diminishing returns formula for the boost
                    consensus_boost = 1.0 + (0.1 * min(num_methods, 3))

                    # Calculate weighted average confidence
                    avg_weighted_confidence = sum(
                        combined[tech_id]["weighted_scores"]
                    ) / len(combined[tech_id]["weighted_scores"])

                    # Apply consensus boost, but cap at 1.0
                    boosted_confidence = min(
                        avg_weighted_confidence * consensus_boost, 1.0
                    )
                    combined[tech_id]["confidence"] = boosted_confidence
                else:
                    # Single method - use weighted confidence
                    combined[tech_id]["confidence"] = weighted_confidence

        # Convert to list and return
        return list(combined.values())

    def extract_techniques_batch(
        self,
        texts: List[str],
        extractors: List[str] = None,
        threshold: float = 0.2,
        top_k: int = 10,
        use_ensemble: bool = True,
        batch_size: int = 5,
        max_workers: int = 2,
    ) -> List[Dict]:
        """
        Extract ATT&CK techniques from multiple texts in batch with parallel processing

        Args:
            texts: List of input texts
            extractors: List of extractors to use
            threshold: Minimum confidence threshold
            top_k: Maximum number of techniques per text
            use_ensemble: Whether to use ensemble method
            batch_size: Size of batches for processing
            max_workers: Maximum number of parallel workers

        Returns:
            List of result dictionaries for each text
        """
        # Process in batches with ThreadPoolExecutor
        results = []

        # Process all texts with single-threaded approach if only a few
        if len(texts) <= batch_size:
            for i, text in enumerate(texts):
                logger.info(f"Processing text {i+1}/{len(texts)}")
                result = self.extract_techniques(
                    text=text,
                    extractors=extractors,
                    threshold=threshold,
                    top_k=top_k,
                    use_ensemble=use_ensemble,
                )
                results.append(result)
            return results

        # Process larger sets in parallel batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                logger.info(
                    f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
                )

                # Submit batch for processing
                futures = []
                for j, text in enumerate(batch):
                    logger.info(f"  Submitting text {i+j+1}/{len(texts)}")
                    future = executor.submit(
                        self.extract_techniques,
                        text=text,
                        extractors=extractors,
                        threshold=threshold,
                        top_k=top_k,
                        use_ensemble=use_ensemble,
                    )
                    futures.append(future)

                # Collect results as they complete
                for future in futures:
                    results.append(future.result())

        return results

    def clear_caches(self):
        """Clear all caches to free memory"""
        for extractor_name, extractor in self.extractors.items():
            if extractor is not None and hasattr(extractor, "clear_cache"):
                try:
                    extractor.clear_cache()
                    logger.info(f"Cleared cache for {extractor_name} extractor")
                except Exception as e:
                    logger.error(f"Error clearing cache for {extractor_name}: {str(e)}")

    # Add these methods to EnhancedATTCKExtractor class

    def process_feedback(
        self,
        analysis_id: str,
        technique_id: str,
        feedback_type: str,
        user_id: str,
        suggested_technique_id: str = None,
        confidence_level: int = None,
        justification_text: str = None,
        highlighted_segments: List[Dict] = None,
    ) -> bool:
        """Process human feedback to improve future extractions

        Args:
            analysis_id: ID of the analysis job
            technique_id: The technique that received feedback
            feedback_type: Type of feedback ('correct', 'incorrect', 'unsure')
            user_id: ID of the user providing feedback
            suggested_technique_id: Alternative technique suggested by user (if any)
            confidence_level: Analyst confidence rating (1-5)
            justification_text: Text explaining the feedback decision
            highlighted_segments: Text segments justifying technique attribution

        Returns:
            Whether feedback was successfully processed
        """
        db = get_db()

        try:
            # Record feedback in database with enhanced fields
            db.execute(
                """
                INSERT INTO analysis_feedback
                (analysis_id, technique_id, user_id, feedback_type, 
                suggested_alternative, confidence_level, justification, 
                created_at) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """,
                (
                    analysis_id,
                    technique_id,
                    user_id,
                    feedback_type,
                    suggested_technique_id,
                    confidence_level,
                    justification_text,
                ),
            )

            # Store highlighted segments if provided
            if highlighted_segments:
                for segment in highlighted_segments:
                    db.execute(
                        """
                        INSERT INTO feedback_highlights
                        (feedback_id, segment_text, start_offset, end_offset)
                        VALUES (LASTVAL(), %s, %s, %s)
                        """,
                        (
                            segment.get("text", ""),
                            segment.get("start", 0),
                            segment.get("end", 0),
                        ),
                    )

            # If feedback is 'incorrect' and there's a suggested alternative,
            # add this to our training data for future model improvements
            if feedback_type == "incorrect" and suggested_technique_id:
                # Get the original text
                job = db.query_one(
                    "SELECT input_data FROM analysis_jobs WHERE id = %s", (analysis_id,)
                )

                if job and job.get("input_data"):
                    # Store as training example (could be used for model fine-tuning)
                    self._store_training_example(
                        job["input_data"],
                        incorrect_technique=technique_id,
                        correct_technique=suggested_technique_id,
                        confidence=confidence_level,
                        justification=justification_text,
                        highlighted_segments=highlighted_segments,
                    )

            return True
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return False

    def _store_training_example(
        self, text: str, incorrect_technique: str, correct_technique: str
    ) -> None:
        """Store feedback as training example for model improvement

        Args:
            text: The input text
            incorrect_technique: Incorrectly identified technique
            correct_technique: Correct technique
        """
        # Create training examples directory if it doesn't exist
        training_dir = os.path.join(self.data_dir, "training_examples")
        os.makedirs(training_dir, exist_ok=True)

        # Store the example
        example = {
            "text": text,
            "incorrect_technique": incorrect_technique,
            "correct_technique": correct_technique,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

        # Generate a unique filename
        filename = os.path.join(
            training_dir,
            f"feedback_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json",
        )

        with open(filename, "w") as f:
            json.dump(example, f, indent=2)

    def get_feedback_for_analysis(self, analysis_id: str, user_id: str) -> list:
        """Retrieve feedback for a specific analysis

        Args:
            analysis_id: ID of the analysis job
            user_id: ID of the user requesting feedback

        Returns:
            List of feedback entries
        """
        try:
            db = get_db()

            # Verify the user has access to this analysis
            job = db.query_one(
                "SELECT user_id FROM analysis_jobs WHERE id = %s", (analysis_id,)
            )

            if not job or job.get("user_id") != user_id:
                logger.warning(
                    f"User {user_id} attempted to access feedback for job {analysis_id}"
                )
                return []

            # Get feedback
            feedback = db.query(
                """
                SELECT id, technique_id, feedback_type, suggested_alternative, created_at
                FROM analysis_feedback 
                WHERE analysis_id = %s
                ORDER BY created_at DESC
                """,
                (analysis_id,),
            )

            return feedback
        except Exception as e:
            logger.error(f"Error retrieving feedback: {str(e)}")
            return []

    def _integrate_heterogeneous_sources(self) -> None:
        """Integrate data from multiple heterogeneous sources"""
        logger.info("Integrating data from heterogeneous sources...")

        # Track source data integrity
        integrity_status = {}

        # 1. Load MITRE ATT&CK data
        logger.info("Loading MITRE ATT&CK framework data...")
        attack_status = self._load_attack_framework_data()
        integrity_status["attack"] = attack_status

        # 2. Load D3FEND data
        logger.info("Loading D3FEND framework data...")
        d3fend_status = self._load_d3fend_data()
        integrity_status["d3fend"] = d3fend_status

        # 3. Load vulnerability data (CVE/KEV)
        logger.info("Loading vulnerability databases...")
        vuln_status = self._load_vulnerability_data()
        integrity_status["vulnerabilities"] = vuln_status

        # 4. Load weakness catalogs (CWE/CAPEC)
        logger.info("Loading weakness catalogs...")
        weakness_status = self._load_weakness_catalogs()
        integrity_status["weaknesses"] = weakness_status

        # 5. Load community contributions if available
        logger.info("Loading community contributions...")
        community_status = self._load_community_contributions()
        integrity_status["community"] = community_status

        # Log integration status
        logger.info(
            f"Data integration status: {json.dumps(integrity_status, indent=2)}"
        )

        # Persist integration status
        if self.use_neo4j:
            self._persist_integration_status(integrity_status)

    def _persist_integration_status(self, status: Dict) -> None:
        """Persist integration status to database

        Args:
            status: Integration status dictionary
        """
        # Create integration status node in Neo4j
        timestamp = datetime.now().isoformat()

        query = """
        CREATE (i:IntegrationStatus {
            timestamp: $timestamp,
            status: $status,
            created_at: datetime()
        })
        """
        params = {"timestamp": timestamp, "status": json.dumps(status)}

        self.neo4j.run_query(query, params)

    def close(self):
        """Close all connections and free resources"""
        # Unload models
        self._unload_memory_intensive_models()

        # Close Neo4j connection
        if self.neo4j:
            self.neo4j.close()

        logger.info("EnhancedATTCKExtractor shut down")


# Singleton instance of extractor
extractor_instance = None


def get_extractor():
    """Get or create the extractor instance"""
    global extractor_instance
    if extractor_instance is None:
        try:
            # Load environment variables or use defaults
            import os

            neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
            neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")
            neo4j_database = os.environ.get("NEO4J_DATABASE", "pipe")
            data_dir = os.environ.get("DATA_DIR", "data")
            models_dir = os.environ.get("MODELS_DIR", "models")
            use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"
            memory_efficient = (
                os.environ.get("MEMORY_EFFICIENT", "true").lower() == "true"
            )

            # Initialize extractor
            extractor_instance = EnhancedATTCKExtractor(
                data_dir=data_dir,
                models_dir=models_dir,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                neo4j_database=neo4j_database,
                use_gpu=use_gpu,
                auto_load=False,  # Don't auto-load - we'll load on demand
                memory_efficient=memory_efficient,
            )
        except Exception as e:
            logger.error(f"Failed to initialize extractor: {str(e)}")
            # Don't raise the exception, just log it and return None
            # This allows the application to start even if the extractor can't be initialized
            return None

    return extractor_instance


# Add model retraining functionality
def queue_model_retraining():
    """Queue a model retraining task based on feedback data"""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis_conn = Redis.from_url(redis_url)
    queue = Queue("model_training", connection=redis_conn)

    # Queue the retraining job
    queue.enqueue(
        extractor_retraining,
        job_timeout="1h",
        result_ttl=86400,
    )


def extractor_retraining():
    """Retrain models based on feedback data"""
    logger.info("Starting model retraining based on feedback data")

    db = get_db()

    # Get positive and negative examples from feedback
    positive_examples = db.query(
        """
        SELECT aj.input_data as text, af.technique_id as technique_id
        FROM analysis_feedback af
        JOIN analysis_jobs aj ON af.analysis_id = aj.id
        WHERE af.feedback_type = 'correct'
        AND af.created_at > (SELECT MAX(last_training_date) FROM model_training_logs)
        """
    )

    negative_examples = db.query(
        """
        SELECT aj.input_data as text, af.technique_id as incorrect_technique,
               af.suggested_alternative as correct_technique
        FROM analysis_feedback af
        JOIN analysis_jobs aj ON af.analysis_id = aj.id
        WHERE af.feedback_type = 'incorrect'
        AND af.suggested_alternative IS NOT NULL
        AND af.created_at > (SELECT MAX(last_training_date) FROM model_training_logs)
        """
    )

    highlighted_examples = db.query(
        """
        SELECT fh.segment_text as text, af.technique_id as technique_id
        FROM feedback_highlights fh
        JOIN analysis_feedback af ON fh.feedback_id = af.id
        WHERE af.feedback_type = 'correct'
        AND af.created_at > (SELECT MAX(last_training_date) FROM model_training_logs)
        """
    )

    # Prepare training data
    training_data = {
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "highlighted_examples": highlighted_examples,
    }

    # Save training data for model update
    training_dir = "data/training"
    os.makedirs(training_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_file = os.path.join(training_dir, f"feedback_training_{timestamp}.json")

    with open(training_file, "w") as f:
        json.dump(training_data, f, indent=2)

    # Record training attempt
    db.execute(
        """
        INSERT INTO model_training_logs 
        (training_data_path, example_count, last_training_date, status)
        VALUES (%s, %s, CURRENT_TIMESTAMP, 'started')
        """,
        (
            training_file,
            len(positive_examples) + len(negative_examples) + len(highlighted_examples),
        ),
    )

    # Train the model (would connect to actual training code)
    try:
        # Placeholder for actual model training code
        logger.info(f"Model retraining would happen here using {training_file}")

        # Update training log status
        db.execute(
            """
            UPDATE model_training_logs
            SET status = 'completed'
            WHERE training_data_path = %s
            """,
            (training_file,),
        )

        return {"status": "success", "training_file": training_file}
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")

        # Update training log status
        db.execute(
            """
            UPDATE model_training_logs
            SET status = 'failed', error_message = %s
            WHERE training_data_path = %s
            """,
            (str(e), training_file),
        )

        return {"status": "error", "message": str(e)}
