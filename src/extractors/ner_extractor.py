"""
Named Entity Recognition Extractor for ATT&CK Techniques
------------------------------------------------------
Uses transformer-based NER models to identify security entities and map them to ATT&CK techniques.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)

from src.database.postgresql import get_db

logger = logging.getLogger("NERExtractor")


class SecureBERTNERExtractor:
    """
    NER extractor using SecureBERT or similar security-focused NER models
    to extract security entities and map them to ATT&CK techniques
    """

    def __init__(
        self,
        model_name: str,
        techniques_data: Dict,
        technique_keywords: Dict,
        cache_dir: str,
        max_length: int = 512,
        use_gpu: bool = True,
        neo4j_connector=None,
    ):
        """
        Initialize NER extractor

        Args:
            model_name: Name or path of the NER model
            techniques_data: Dictionary of technique data
            technique_keywords: Dictionary of technique keywords
            cache_dir: Directory for model caching
            max_length: Maximum sequence length for tokenization
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.techniques_data = techniques_data
        self.technique_keywords = technique_keywords
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.neo4j_connector = neo4j_connector

        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.device = "cuda" if self.use_gpu else "cpu"

        # Status flags
        self.is_loaded = False

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Log initialization
        logger.info(f"Initialized NER extractor with model {model_name}")
        logger.info(f"Using device: {self.device}")

    def load_model(self) -> bool:
        """
        Load the NER model and tokenizer

        Returns:
            Whether loading was successful
        """
        if self.is_loaded:
            return True

        try:
            # First check if we have a local copy in the cache directory
            if os.path.exists(self.cache_dir):
                # Check if model files exist
                config_file = os.path.join(self.cache_dir, "config.json")
                if os.path.exists(config_file):
                    logger.info(f"Loading NER model from local cache: {self.cache_dir}")
                    try:
                        # Try loading from cache directory
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.cache_dir, use_fast=True, local_files_only=True
                        )

                        self.model = AutoModelForTokenClassification.from_pretrained(
                            self.cache_dir, local_files_only=True
                        )

                        # Move to device
                        self.model = self.model.to(self.device)

                        # Create NER pipeline
                        self.ner_pipeline = pipeline(
                            "ner",
                            model=self.model,
                            tokenizer=self.tokenizer,
                            device=0 if self.use_gpu else -1,
                            aggregation_strategy="simple",
                        )

                        self.is_loaded = True
                        logger.info(
                            f"NER model loaded successfully from local cache on {self.device}"
                        )
                        return True
                    except Exception as e:
                        logger.warning(
                            f"Error loading from cache, will try downloading: {str(e)}"
                        )
                        # Continue to download attempt

            # If we reach here, try downloading from Hugging Face
            logger.info(
                f"Downloading NER model from Hugging Face: CyberPeace-Institute/SecureBERT-NER"
            )
            # Set a reasonable timeout for model download
            download_timeout = 60  # seconds

            try:
                # Load tokenizer with timeout
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "CyberPeace-Institute/SecureBERT-NER",
                    cache_dir=self.cache_dir,
                    use_fast=True,
                    local_files_only=False,
                )

                # Load model with timeout
                self.model = AutoModelForTokenClassification.from_pretrained(
                    "CyberPeace-Institute/SecureBERT-NER",
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                )

                # Move model to device
                self.model = self.model.to(self.device)

                # Create NER pipeline
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.use_gpu else -1,
                    aggregation_strategy="simple",
                )

                # Save model locally for future use
                self.tokenizer.save_pretrained(self.cache_dir)
                self.model.save_pretrained(self.cache_dir)
                logger.info(f"Saved model to cache directory: {self.cache_dir}")

                self.is_loaded = True
                logger.info(
                    f"NER model downloaded and loaded successfully on {self.device}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to download and load NER model: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Failed to load NER model: {str(e)}")
            return False

    def unload_model(self) -> None:
        """Unload the model to free memory"""
        if not self.is_loaded:
            return

        try:
            # Delete model, tokenizer, and pipeline
            del self.model
            del self.tokenizer
            del self.ner_pipeline

            # Set to None
            self.model = None
            self.tokenizer = None
            self.ner_pipeline = None

            # Force garbage collection
            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_loaded = False
            logger.info("NER model unloaded")

        except Exception as e:
            logger.error(f"Error unloading NER model: {str(e)}")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text

        Args:
            text: Input text

        Returns:
            Dictionary mapping entity types to lists of entities
        """
        if not self.is_loaded:
            if not self.load_model():
                logger.error("NER model not loaded")
                return {}

        try:
            # Process the text in chunks if it's too long
            max_chars = 10000  # Reasonable limit to avoid memory issues

            if len(text) > max_chars:
                logger.info(
                    f"Text is too long ({len(text)} chars), processing first {max_chars} chars"
                )
                text = text[:max_chars]

            # Get entities
            entities = self.ner_pipeline(text)

            # Group by entity type
            grouped_entities = {}

            for entity in entities:
                entity_type = entity.get("entity_group", "")
                entity_text = entity.get("word", "")

                if not entity_type or not entity_text:
                    continue

                if entity_type not in grouped_entities:
                    grouped_entities[entity_type] = []

                # Add if not already in list
                if entity_text not in grouped_entities[entity_type]:
                    grouped_entities[entity_type].append(entity_text)

            return grouped_entities

        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {}

    # Modify the extract_techniques method in src/extractors/ner_extractor.py
    def extract_techniques(
        self,
        text: str,
        min_confidence: float = 0.1,
        max_results: int = 10,
        job_id: str = None,
    ) -> List[Dict]:
        """
        Extract techniques using NER entity matching with metrics recording

        Args:
            text: Input text
            min_confidence: Minimum confidence threshold
            max_results: Maximum number of results
            job_id: Optional job ID for metrics recording

        Returns:
            List of technique matches with confidence scores
        """
        # Create metrics recorder if job_id is provided
        metrics_recorder = None
        if job_id:
            from src.database.metrics_recorder import MetricsRecorder

            metrics_recorder = MetricsRecorder(job_id)

        # Record start time for performance metrics
        start_time = time.time()

        # Record extractor execution details if metrics enabled
        extractor_id = None
        if metrics_recorder:
            parameters = {
                "min_confidence": min_confidence,
                "max_results": max_results,
                "model_name": self.model_name,
                "device": self.device,
                "max_length": self.max_length,
            }
            extractor_id = metrics_recorder.record_extractor_result(
                extractor_name="ner",
                raw_input=text,
                raw_output={},  # Will update this later
                execution_time_ms=0,  # Will update this later
                parameters=parameters,
            )

            # Record model details if available
            if hasattr(self, "model") and self.model is not None:
                # Record model weights
                metrics_recorder.record_model_weights(
                    extractor_id=extractor_id,
                    model_name=self.model_name,
                    weight_path=self.cache_dir,
                )

        # Load model if needed
        load_start = time.time()
        if not self.is_loaded:
            self.load_model()

            # Record model loading performance
            if metrics_recorder:
                load_time_ms = int((time.time() - load_start) * 1000)
                metrics_recorder.record_performance_benchmark(
                    extractor_name="ner",
                    operation_type="model_loading",
                    execution_time_ms=load_time_ms,
                )

        # Extract entities
        entity_start = time.time()
        entities = self.extract_entities(text)
        entity_time_ms = int((time.time() - entity_start) * 1000)

        if not entities:
            logger.info("No entities found in text")

            # Update extractor result with empty output and execution time
            if metrics_recorder and extractor_id:
                query = """
                UPDATE extractor_results
                SET raw_output = %s, execution_time_ms = %s
                WHERE id = %s
                """

                from psycopg2.extras import Json

                get_db().execute(
                    query,
                    (Json([]), int((time.time() - start_time) * 1000), extractor_id),
                )

            return []

        # Record NER details if metrics enabled
        if metrics_recorder and extractor_id:
            # Count entity types
            entity_type_counts = {}
            for entity_type, entity_list in entities.items():
                entity_type_counts[entity_type] = len(entity_list)

            total_entity_count = sum(len(entities) for entities in entities.values())

            # Record NER details
            metrics_recorder.record_ner_details(
                extractor_id=extractor_id,
                entity_count=total_entity_count,
                entity_types=entity_type_counts,
                model_name=self.model_name,
                aggregation_strategy="simple",  # Common value for NER extractors
                tokenizer_max_length=self.max_length,
            )

            # Record entity extraction performance
            metrics_recorder.record_performance_benchmark(
                extractor_name="ner",
                operation_type="entity_extraction",
                execution_time_ms=entity_time_ms,
                input_size=len(text),
            )

            # Record each entity found
            for entity_type, entity_list in entities.items():
                for i, entity_text in enumerate(entity_list):
                    # Find position in text
                    pos = text.lower().find(entity_text.lower())

                    metrics_recorder.record_entities(
                        extractor_id=extractor_id,
                        entities=[
                            {
                                "text": entity_text,
                                "type": entity_type,
                                "start_offset": pos if pos >= 0 else None,
                                "end_offset": pos + len(entity_text)
                                if pos >= 0
                                else None,
                                "confidence": None,  # NER pipeline doesn't always provide per-entity confidence
                            }
                        ],
                    )

        # Get all entities as a flat list
        all_entities = []
        for entity_type, entity_list in entities.items():
            all_entities.extend(entity_list)

        logger.info(
            f"Found {len(all_entities)} entities across {len(entities)} entity types"
        )

        # Match entities to techniques
        technique_matches = {}

        match_start = time.time()
        for entity in all_entities:
            entity_lower = entity.lower()

            # Check for direct matches in technique keywords
            for tech_id, keywords in self.technique_keywords.items():
                for keyword in keywords:
                    keyword_lower = keyword.lower()

                    # Check if entity contains keyword or vice versa
                    if keyword_lower in entity_lower or entity_lower in keyword_lower:
                        if tech_id not in technique_matches:
                            technique_matches[tech_id] = {"count": 0, "entities": []}

                        technique_matches[tech_id]["count"] += 1
                        if entity not in technique_matches[tech_id]["entities"]:
                            technique_matches[tech_id]["entities"].append(entity)

        match_time_ms = int((time.time() - match_start) * 1000)

        # Record technique matching performance
        if metrics_recorder and extractor_id:
            metrics_recorder.record_performance_benchmark(
                extractor_name="ner",
                operation_type="technique_matching",
                execution_time_ms=match_time_ms,
                input_size=len(all_entities),
            )

        # Convert matches to results
        results = []

        for tech_id, match_data in technique_matches.items():
            # Calculate confidence based on match count and unique entities
            match_count = match_data["count"]
            unique_entities = len(match_data["entities"])

            # Confidence formula: base + boost for multiple entities
            confidence = min(0.4 + (unique_entities * 0.05), 0.85)

            result = {
                "technique_id": tech_id,
                "confidence": confidence,
                "match_count": match_count,
                "matched_entities": match_data["entities"],
                "method": "ner",
            }

            results.append(result)

        # Filter by confidence threshold
        results = [r for r in results if r["confidence"] >= min_confidence]

        # Sort by confidence and limit results
        results.sort(key=lambda x: x["confidence"], reverse=True)
        results = results[:max_results]

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Update extractor result with final output and execution time
        if metrics_recorder and extractor_id:
            query = """
            UPDATE extractor_results
            SET raw_output = %s, execution_time_ms = %s
            WHERE id = %s
            """

            from psycopg2.extras import Json

            get_db().execute(query, (Json(results), execution_time_ms, extractor_id))

        return results

    def extract_cve_entities(self, text: str) -> List[str]:
        """
        Extract CVE identifiers from text

        Args:
            text: Input text

        Returns:
            List of CVE identifiers
        """
        # Extract all entities
        entities = self.extract_entities(text)

        # Look for CVE entities
        cve_entities = []

        # Check "VULNERABILITY" type if present
        if "VULNERABILITY" in entities:
            for entity in entities["VULNERABILITY"]:
                if "CVE-" in entity:
                    cve_entities.append(entity)

        # Check all entity types for CVE pattern
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if "CVE-" in entity and entity not in cve_entities:
                    cve_entities.append(entity)

        return cve_entities
