"""
Classifier for ATT&CK Techniques
------------------------------
Provides classification and ensemble logic for ATT&CK technique identification.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer

from src.database.postgresql import get_db

logger = logging.getLogger("ClassifierExtractor")


class SecureBERTClassifier:
    """
    Classifier for ATT&CK techniques using transformer models
    Also provides ensemble functionality to combine results from multiple extractors
    """

    def __init__(
        self,
        techniques_data: Dict,
        cache_dir: str,
        embeddings_dir: str = None,
        base_model: str = "ehsanaghaei/SecureBERT",
        use_gpu: bool = True,
        neo4j_connector=None,
    ):
        """
        Initialize classifier

        Args:
            techniques_data: Dictionary of technique data
            cache_dir: Directory for model caching
            embeddings_dir: Directory for embeddings storage
            base_model: Base transformer model to use
            use_gpu: Whether to use GPU acceleration
        """
        self.techniques_data = techniques_data
        self.cache_dir = cache_dir
        self.embeddings_dir = embeddings_dir or os.path.join(cache_dir, "embeddings")
        self.base_model = base_model
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.neo4j_connector = neo4j_connector

        # Create directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # Paths for model storage
        self.base_model_dir = os.path.join(cache_dir, "base_model")
        self.classifier_path = os.path.join(cache_dir, "classifier.joblib")

        # Model components
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.device = "cuda" if self.use_gpu else "cpu"

        # Flags
        self.base_model_loaded = False
        self.classifier_loaded = False

        # Log initialization
        logger.info(f"Initialized classifier with base model {base_model}")
        logger.info(f"Using device: {self.device}")

    def load_base_model(self) -> bool:
        """
        Load the base transformer model

        Returns:
            Whether loading was successful
        """
        if self.base_model_loaded:
            return True

        try:
            # First check if we have a local copy in the base_model_dir
            if os.path.exists(self.base_model_dir):
                # Check if model files exist
                config_file = os.path.join(self.base_model_dir, "config.json")
                if os.path.exists(config_file):
                    logger.info(
                        f"Loading base model from local cache: {self.base_model_dir}"
                    )
                    try:
                        # Try loading from cache directory
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.base_model_dir, use_fast=True, local_files_only=True
                        )

                        self.model = AutoModel.from_pretrained(
                            self.base_model_dir, local_files_only=True
                        )

                        # Move to device
                        self.model = self.model.to(self.device)

                        self.base_model_loaded = True
                        logger.info(
                            f"Base model loaded successfully from local cache on {self.device}"
                        )
                        return True
                    except Exception as e:
                        logger.warning(
                            f"Error loading from cache, will try downloading: {str(e)}"
                        )
                        # Continue to download attempt

            # If we reach here, try downloading from Hugging Face
            logger.info(f"Downloading base model from Hugging Face: {self.base_model}")

            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model,
                    cache_dir=self.cache_dir,
                    use_fast=True,
                    local_files_only=False,
                )

                # Load model
                self.model = AutoModel.from_pretrained(
                    self.base_model, cache_dir=self.cache_dir, local_files_only=False
                )

                # Move to device
                self.model = self.model.to(self.device)

                # Save for future use
                os.makedirs(self.base_model_dir, exist_ok=True)
                self.tokenizer.save_pretrained(self.base_model_dir)
                self.model.save_pretrained(self.base_model_dir)
                logger.info(
                    f"Saved model to base model directory: {self.base_model_dir}"
                )

                self.base_model_loaded = True
                logger.info(
                    f"Base model downloaded and loaded successfully on {self.device}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to download and load base model: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Failed to load base model: {str(e)}")
            return False

    def load_classifier(self) -> bool:
        """
        Load the classifier model

        Returns:
            Whether loading was successful
        """
        if self.classifier_loaded:
            return True

        if os.path.exists(self.classifier_path):
            try:
                # Load trained classifier
                self.classifier = joblib.load(self.classifier_path)

                self.classifier_loaded = True
                logger.info("Classifier loaded successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to load classifier: {str(e)}")

        # Initialize default classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )

        logger.info("Initialized default classifier (not trained)")
        return False

    def unload_models(self) -> None:
        """Unload models to free memory"""
        if self.base_model_loaded:
            try:
                # Delete model and tokenizer
                del self.model
                del self.tokenizer

                # Set to None
                self.model = None
                self.tokenizer = None

                # Force garbage collection
                import gc

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.base_model_loaded = False
                logger.info("Base model unloaded")

            except Exception as e:
                logger.error(f"Error unloading base model: {str(e)}")

        if self.classifier_loaded:
            try:
                # Delete classifier
                del self.classifier

                # Set to None
                self.classifier = None

                # Force garbage collection
                import gc

                gc.collect()

                self.classifier_loaded = False
                logger.info("Classifier unloaded")

            except Exception as e:
                logger.error(f"Error unloading classifier: {str(e)}")

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using the base model

        Args:
            text: Input text

        Returns:
            Text embedding
        """
        if not self.base_model_loaded:
            if not self.load_base_model():
                logger.error("Base model not loaded")
                return np.zeros(768)  # Default embedding dimension

        try:
            # Tokenize
            inputs = self.tokenizer(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embedding
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            return embedding[0]

        except Exception as e:
            logger.error(f"Error getting text embedding: {str(e)}")
            return np.zeros(768)  # Default embedding dimension

    # Modify the classify_techniques method in src/extractors/classifier.py
    def classify_techniques(
        self, text: str, threshold: float = 0.2, top_k: int = 10, job_id: str = None
    ) -> List[Dict]:
        """
        Classify ATT&CK techniques directly from text with metrics recording

        Args:
            text: Input text
            threshold: Minimum confidence threshold
            top_k: Maximum number of results
            job_id: Optional job ID for metrics recording

        Returns:
            List of technique classifications with confidence scores
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
                "threshold": threshold,
                "top_k": top_k,
                "model_name": self.base_model,
                "device": self.device,
            }
            extractor_id = metrics_recorder.record_extractor_result(
                extractor_name="classifier",
                raw_input=text,
                raw_output={},  # Will update this later
                execution_time_ms=0,  # Will update this later
                parameters=parameters,
            )

        # Get text embedding
        embed_start = time.time()
        text_embedding = self._get_text_embedding(text)
        embed_time_ms = int((time.time() - embed_start) * 1000)

        # Record embedding performance
        if metrics_recorder and extractor_id:
            # Record text segment with embedding
            segment_id = metrics_recorder.record_text_segment(
                text=text, index=0, embedding=text_embedding
            )

            # Record embedding details
            metrics_recorder.record_embedding_details(
                extractor_id=extractor_id,
                text_segment_id=segment_id,
                embedding_type="classifier_input",
                embedding_model=self.base_model,
                embedding_dimension=len(text_embedding),
                normalization_applied=True,
            )

            # Record performance benchmark
            metrics_recorder.record_performance_benchmark(
                extractor_name="classifier",
                operation_type="text_embedding",
                execution_time_ms=embed_time_ms,
                input_size=len(text),
            )

        # If classifier not trained, use embedding similarity
        if not self.classifier_loaded or self.classifier is None:
            logger.warning("Classifier not trained, using embedding similarity instead")
            results = self._similar_techniques_by_embedding(
                text_embedding, threshold, top_k
            )

            # Record execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Update extractor result
            if metrics_recorder and extractor_id:
                query = """
                UPDATE extractor_results
                SET raw_output = %s, execution_time_ms = %s
                WHERE id = %s
                """

                from psycopg2.extras import Json

                get_db().execute(
                    query, (Json(results), execution_time_ms, extractor_id)
                )

            return results

        try:
            # Prepare features for classifier
            features = text_embedding.reshape(1, -1)

            # Get prediction probabilities
            classify_start = time.time()
            probas = self.classifier.predict_proba(features)
            classify_time_ms = int((time.time() - classify_start) * 1000)

            # Record classifier details
            if metrics_recorder and extractor_id:
                # Record classifier performance
                metrics_recorder.record_performance_benchmark(
                    extractor_name="classifier",
                    operation_type="classification",
                    execution_time_ms=classify_time_ms,
                )

                # Record classifier details
                probability_scores = {}
                for i, class_label in enumerate(self.classifier.classes_):
                    if i < len(probas[0]):
                        probability_scores[class_label] = float(probas[0][i])

                metrics_recorder.record_classifier_details(
                    extractor_id=extractor_id,
                    model_type="RandomForest"
                    if isinstance(self.classifier, RandomForestClassifier)
                    else "Unknown",
                    feature_count=text_embedding.shape[0],
                    class_count=len(self.classifier.classes_),
                    probability_scores=probability_scores,
                    decision_threshold=threshold,
                    embedding_used=True,
                )

            # Create results
            results = []

            for i, tech_id in enumerate(self.classifier.classes_):
                if i < len(probas[0]):
                    confidence = probas[0][i]

                    if confidence >= threshold:
                        result = {
                            "technique_id": tech_id,
                            "confidence": float(confidence),
                            "method": "classifier",
                        }

                        results.append(result)

            # Sort by confidence and limit results
            results.sort(key=lambda x: x["confidence"], reverse=True)
            results = results[:top_k]

            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Update extractor result
            if metrics_recorder and extractor_id:
                query = """
                UPDATE extractor_results
                SET raw_output = %s, execution_time_ms = %s
                WHERE id = %s
                """

                from psycopg2.extras import Json

                get_db().execute(
                    query, (Json(results), execution_time_ms, extractor_id)
                )

            return results

        except Exception as e:
            logger.error(f"Error classifying techniques: {str(e)}")

            # Record error
            if metrics_recorder and extractor_id:
                query = """
                UPDATE extractor_results
                SET raw_output = %s, execution_time_ms = %s
                WHERE id = %s
                """

                from psycopg2.extras import Json

                get_db().execute(
                    query,
                    (
                        Json({"error": str(e)}),
                        int((time.time() - start_time) * 1000),
                        extractor_id,
                    ),
                )

            return []

    def _similar_techniques_by_embedding(
        self, embedding: np.ndarray, threshold: float = 0.5, top_k: int = 10
    ) -> List[Dict]:
        """
        Find similar techniques using embedding similarity

        Args:
            embedding: Query embedding
            threshold: Minimum similarity threshold
            top_k: Maximum number of results

        Returns:
            List of technique matches with scores
        """
        # Check for cached technique embeddings
        embeddings_file = os.path.join(self.embeddings_dir, "technique_embeddings.npy")
        embedding_ids_file = os.path.join(
            self.embeddings_dir, "technique_embedding_ids.json"
        )

        if not os.path.exists(embeddings_file) or not os.path.exists(
            embedding_ids_file
        ):
            logger.warning(
                "No technique embeddings found, cannot perform similarity search"
            )
            return []

        try:
            # Load embeddings
            technique_embeddings = np.load(embeddings_file)

            with open(embedding_ids_file, "r") as f:
                technique_ids = json.load(f)

            # Calculate similarities
            similarities = []

            for i, tech_id in enumerate(technique_ids):
                if i >= len(technique_embeddings):
                    continue

                tech_embedding = technique_embeddings[i]

                # Calculate cosine similarity
                similarity = self._cosine_similarity(embedding, tech_embedding)

                similarities.append((tech_id, similarity))

            # Filter by threshold
            filtered_similarities = [
                (tech_id, sim) for tech_id, sim in similarities if sim >= threshold
            ]

            # Sort and limit
            filtered_similarities.sort(key=lambda x: x[1], reverse=True)
            filtered_similarities = filtered_similarities[:top_k]

            # Convert to results
            results = []

            for tech_id, similarity in filtered_similarities:
                result = {
                    "technique_id": tech_id,
                    "confidence": similarity,
                    "method": "embedding_similarity",
                }

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error finding similar techniques: {str(e)}")
            return []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0

        similarity = np.dot(a, b) / (norm_a * norm_b)

        # Ensure result is in valid range
        return max(0.0, min(1.0, similarity))

    def ensemble_extractors(
        self,
        text: str,
        rule_results: List[Dict] = None,
        bm25_results: List[Dict] = None,
        ner_results: List[Dict] = None,
        threshold: float = 0.2,
        max_results: int = 10,
        kev_results: List[Dict] = None,  # Added parameter for KEV results
        semantic_results: List[Dict] = None,  # Added parameter for semantic results
    ) -> List[Dict]:
        """
        Combine results from multiple extractors with smart ensemble logic

        Args:
            text: Input text
            rule_results: Results from rule-based extractor
            bm25_results: Results from BM25 extractor
            ner_results: Results from NER extractor
            threshold: Minimum confidence threshold
            max_results: Maximum number of results
            kev_results: Results from KEV extractor
            semantic_results: Results from semantic extractor

        Returns:
            List of ensemble results with confidence scores
        """
        # Ensure we have at least one set of results
        if (
            not rule_results
            and not bm25_results
            and not ner_results
            and not kev_results
            and not semantic_results
        ):
            logger.warning("No extractor results provided for ensemble")
            return []

        # Get all unique technique IDs from results
        technique_ids = set()

        if rule_results:
            technique_ids.update(r["technique_id"] for r in rule_results)

        if bm25_results:
            technique_ids.update(r["technique_id"] for r in bm25_results)

        if ner_results:
            technique_ids.update(r["technique_id"] for r in ner_results)

        if kev_results:
            technique_ids.update(r["technique_id"] for r in kev_results)

        if semantic_results:
            technique_ids.update(r["technique_id"] for r in semantic_results)

        # Get text embedding for verification
        text_embedding = self._get_text_embedding(text)

        # Process each technique
        ensemble_results = []

        for tech_id in technique_ids:
            # Get scores from each extractor
            rule_score = (
                next(
                    (
                        r["confidence"]
                        for r in rule_results
                        if r["technique_id"] == tech_id
                    ),
                    0.0,
                )
                if rule_results
                else 0.0
            )
            bm25_score = (
                next(
                    (
                        r["confidence"]
                        for r in bm25_results
                        if r["technique_id"] == tech_id
                    ),
                    0.0,
                )
                if bm25_results
                else 0.0
            )
            ner_score = (
                next(
                    (
                        r["confidence"]
                        for r in ner_results
                        if r["technique_id"] == tech_id
                    ),
                    0.0,
                )
                if ner_results
                else 0.0
            )
            kev_score = (
                next(
                    (
                        r["confidence"]
                        for r in kev_results
                        if r["technique_id"] == tech_id
                    ),
                    0.0,
                )
                if kev_results
                else 0.0
            )
            semantic_score = (
                next(
                    (
                        r["confidence"]
                        for r in semantic_results
                        if r["technique_id"] == tech_id
                    ),
                    0.0,
                )
                if semantic_results
                else 0.0
            )

            # Count non-zero scores
            non_zero_scores = sum(
                score > 0
                for score in [
                    rule_score,
                    bm25_score,
                    ner_score,
                    kev_score,
                    semantic_score,
                ]
            )

            if non_zero_scores == 0:
                continue

            # Calculate weighted average
            # Give more weight to extractors with higher confidence
            weights = {
                "rule": 0.25,
                "bm25": 0.2,
                "ner": 0.2,
                "kev": 0.25,
                "semantic": 0.1,
            }

            weighted_sum = (
                rule_score * weights["rule"]
                + bm25_score * weights["bm25"]
                + ner_score * weights["ner"]
                + kev_score * weights["kev"]
                + semantic_score * weights["semantic"]
            )

            # Apply confirmation boost when multiple extractors agree
            boost_factor = 1.0
            if non_zero_scores > 1:
                # Boost confidence when multiple extractors agree
                boost_factor = 1.0 + (0.1 * non_zero_scores)

            # Additional boost for KEV-identified techniques (high reliability)
            if kev_score > 0:
                boost_factor *= 1.1

            # Calculate ensemble confidence
            ensemble_confidence = min(weighted_sum * boost_factor, 1.0)

            # Skip low confidence results
            if ensemble_confidence < threshold:
                continue

            # Create ensemble result
            ensemble_result = {
                "technique_id": tech_id,
                "confidence": ensemble_confidence,
                "method": "ensemble",
                "component_scores": {
                    "rule_based": rule_score,
                    "bm25": bm25_score,
                    "ner": ner_score,
                    "kev": kev_score,
                    "semantic": semantic_score,
                },
            }

            # Preserve relevant fields from extractors

            # Get technique name from any extractor that provided it
            for results in [
                rule_results,
                bm25_results,
                ner_results,
                kev_results,
                semantic_results,
            ]:
                if not results:
                    continue

                for result in results:
                    if result.get("technique_id") == tech_id and "name" in result:
                        ensemble_result["name"] = result["name"]
                        break

            # Preserve matched keywords if available
            for results in [rule_results, bm25_results]:
                if not results:
                    continue

                for result in results:
                    if (
                        result.get("technique_id") == tech_id
                        and "matched_keywords" in result
                    ):
                        ensemble_result["matched_keywords"] = result["matched_keywords"]
                        break

            # Preserve CVE info if available
            if kev_results:
                for result in kev_results:
                    if result.get("technique_id") == tech_id and "cve_id" in result:
                        ensemble_result["cve_id"] = result["cve_id"]
                        break

            ensemble_results.append(ensemble_result)

        # Sort by confidence and limit results
        ensemble_results.sort(key=lambda x: x["confidence"], reverse=True)
        return ensemble_results[:max_results]
