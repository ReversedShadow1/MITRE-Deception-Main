"""
BM25 Extractor for ATT&CK Techniques
-----------------------------------
Implements BM25 ranking algorithm for identifying ATT&CK techniques.
"""

import json
import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus

from src.database.postgresql import get_db

logger = logging.getLogger("BM25Extractor")


class BM25Extractor:
    """BM25-based extractor for ATT&CK techniques"""

    def __init__(
        self,
        techniques: Dict,
        technique_keywords: Dict,
        models_dir: str = "models/bm25",
        bm25_variant: str = "plus",
        use_cache: bool = True,
        neo4j_connector=None,
    ):
        """
        Initialize BM25 extractor

        Args:
            techniques: Dictionary of technique data
            technique_keywords: Dictionary of technique keywords
            models_dir: Directory for model storage
            bm25_variant: BM25 variant to use ('okapi' or 'plus')
            use_cache: Whether to use cached model
        """
        self.techniques = techniques
        self.technique_keywords = technique_keywords
        self.models_dir = models_dir
        self.bm25_variant = bm25_variant.lower()
        self.use_cache = use_cache
        self.neo4j_connector = neo4j_connector

        # Create model directory
        os.makedirs(models_dir, exist_ok=True)

        # Set cache file paths
        self.corpus_cache_path = os.path.join(models_dir, "corpus.json")
        self.model_cache_path = os.path.join(
            models_dir, f"bm25_{bm25_variant}_model.pkl"
        )

        # Initialize model components
        self.corpus = []
        self.tokenized_corpus = []
        self.tech_ids = []
        self.bm25_model = None

        # Auto-initialize
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the BM25 model"""
        # Try loading from cache first
        if self.use_cache and self._load_cached_model():
            return

        # Build from scratch otherwise
        self._build_model()

    def _load_cached_model(self) -> bool:
        """
        Load model from cache

        Returns:
            Whether load was successful
        """
        if not os.path.exists(self.corpus_cache_path) or not os.path.exists(
            self.model_cache_path
        ):
            return False

        try:
            # Load corpus and technique IDs
            with open(self.corpus_cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                self.corpus = cache_data.get("corpus", [])
                self.tech_ids = cache_data.get("tech_ids", [])
                self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]

            # Load BM25 model
            with open(self.model_cache_path, "rb") as f:
                self.bm25_model = pickle.load(f)

            logger.info(
                f"Loaded BM25 model from cache with {len(self.corpus)} documents"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load cached BM25 model: {str(e)}")
            return False

    def _build_model(self) -> None:
        """Build BM25 model from technique data"""
        logger.info("Building BM25 model from scratch...")

        self.corpus = []
        self.tech_ids = []

        # Build corpus from technique data and keywords
        for tech_id, tech_data in self.techniques.items():
            # Skip techniques without keywords
            if tech_id not in self.technique_keywords:
                continue

            # Get technique data
            tech_name = tech_data.get("name", "")
            tech_desc = tech_data.get("description", "")
            tech_keywords = self.technique_keywords.get(tech_id, [])

            # Combine name, description, and keywords
            doc_parts = [tech_name]

            # Add first sentence of description
            if tech_desc:
                first_sentence = tech_desc.split(".")[0]
                doc_parts.append(first_sentence)

            # Add all keywords
            doc_parts.extend(tech_keywords)

            # Join all parts
            doc = " ".join(doc_parts)

            # Add to corpus
            self.corpus.append(doc)
            self.tech_ids.append(tech_id)

        # Tokenize corpus
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]

        # Create BM25 model
        if self.bm25_variant == "plus":
            self.bm25_model = BM25Plus(self.tokenized_corpus)
        else:
            self.bm25_model = BM25Okapi(self.tokenized_corpus)

        logger.info(f"Built BM25 model with {len(self.corpus)} documents")

        # Cache model if enabled
        if self.use_cache:
            self._cache_model()

    def _cache_model(self) -> None:
        """Cache BM25 model to disk"""
        try:
            # Save corpus and tech IDs
            with open(self.corpus_cache_path, "w", encoding="utf-8") as f:
                cache_data = {"corpus": self.corpus, "tech_ids": self.tech_ids}
                json.dump(cache_data, f)

            # Save BM25 model
            with open(self.model_cache_path, "wb") as f:
                pickle.dump(self.bm25_model, f)

            logger.info("Cached BM25 model to disk")

        except Exception as e:
            logger.error(f"Failed to cache BM25 model: {str(e)}")

    # Modify the extract_techniques method in src/extractors/bm25_extractor.py
    def extract_techniques(
        self, text: str, threshold: float = 0.1, top_k: int = 10, job_id: str = None
    ) -> List[Dict]:
        """
        Extract techniques using BM25 ranking with metrics recording

        Args:
            text: Input text to analyze
            threshold: Minimum score threshold (0-1)
            top_k: Maximum number of results
            job_id: Optional job ID for metrics recording

        Returns:
            List of technique matches with scores
        """
        if not self.bm25_model:
            logger.error("BM25 model not initialized")
            return []

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
                "threshold": threshold,
                "top_k": top_k,
                "bm25_variant": self.bm25_variant,
            }
            extractor_id = metrics_recorder.record_extractor_result(
                extractor_name="bm25",
                raw_input=text,
                raw_output={},  # Will update this later
                execution_time_ms=0,  # Will update this later
                parameters=parameters,
            )

        # Tokenize query
        query_tokens = text.lower().split()

        # Get BM25 scores
        tokenize_start = time.time()
        scores = self.bm25_model.get_scores(query_tokens)
        tokenize_time_ms = int((time.time() - tokenize_start) * 1000)

        # Record tokenization performance
        if metrics_recorder and extractor_id:
            metrics_recorder.record_performance_benchmark(
                extractor_name="bm25",
                operation_type="tokenization",
                execution_time_ms=tokenize_time_ms,
                input_size=len(text),
            )

        # Find max score for normalization - FIX: properly check if scores array is empty
        max_score = max(scores) if len(scores) > 0 else 1.0

        # Convert to results
        results = []

        for i, score in enumerate(scores):
            # Skip if index out of range
            if i >= len(self.tech_ids):
                continue

            # Skip low scores
            normalized_score = score / max_score if max_score > 0 else 0

            if normalized_score >= threshold:
                tech_id = self.tech_ids[i]

                # Create result
                result = {
                    "technique_id": tech_id,
                    "confidence": float(
                        normalized_score
                    ),  # Convert numpy float to Python float
                    "raw_score": float(
                        score
                    ),  # Convert to float for JSON serialization
                    "method": "bm25",
                }

                # Record BM25 score details
                if metrics_recorder and extractor_id:
                    # Get matching terms from the document
                    matched_terms = {}
                    if i < len(self.tokenized_corpus):
                        doc_tokens = self.tokenized_corpus[i]
                        for token in query_tokens:
                            if token in doc_tokens:
                                matched_terms[token] = doc_tokens.count(token)

                    metrics_recorder.record_bm25_scores(
                        extractor_id=extractor_id,
                        technique_id=tech_id,
                        raw_score=float(score),
                        normalized_score=float(normalized_score),
                        matched_terms=matched_terms,
                    )

                results.append(result)

        # Sort by confidence and limit to top_k
        results.sort(key=lambda x: x["confidence"], reverse=True)
        results = results[:top_k]

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

    def clear_cache(self) -> None:
        """Clear the model cache"""
        if os.path.exists(self.corpus_cache_path):
            try:
                os.remove(self.corpus_cache_path)
            except Exception as e:
                logger.error(f"Failed to remove corpus cache: {str(e)}")

        if os.path.exists(self.model_cache_path):
            try:
                os.remove(self.model_cache_path)
            except Exception as e:
                logger.error(f"Failed to remove model cache: {str(e)}")
