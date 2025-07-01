"""
Semantic Embedding Extractor for ATT&CK Techniques
------------------------------------------------
Uses embedding models to perform semantic search for ATT&CK techniques.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.database.postgresql import get_db

logger = logging.getLogger("SemanticExtractor")


class BGEEmbeddingExtractor:
    """
    Semantic search extractor using BGE or similar embedding models
    to find semantically similar ATT&CK techniques
    """

    def __init__(
        self,
        model_name: str,
        techniques: Dict,
        technique_keywords: Dict,
        models_dir: str,
        cache_embeddings: bool = True,
        use_gpu: bool = True,
        neo4j_connector=None,
    ):
        """
        Initialize semantic extractor

        Args:
            model_name: Name or path of the embedding model
            techniques: Dictionary of technique data
            technique_keywords: Dictionary of technique keywords
            models_dir: Directory for model storage
            cache_embeddings: Whether to cache technique embeddings
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.techniques = techniques
        self.technique_keywords = technique_keywords
        self.models_dir = models_dir
        self.cache_embeddings = cache_embeddings
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.neo4j_connector = neo4j_connector

        # Model components
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if self.use_gpu else "cpu"

        # Embedding cache
        self.embeddings_cache_path = os.path.join(
            models_dir, "technique_embeddings.npy"
        )
        self.embedding_ids_path = os.path.join(
            models_dir, "technique_embedding_ids.json"
        )
        self.technique_embeddings = None
        self.embedding_tech_ids = []

        # Status flags
        self.is_loaded = False

        # Create model directory
        os.makedirs(models_dir, exist_ok=True)

        # Log initialization
        logger.info(f"Initialized semantic extractor with model {model_name}")
        logger.info(f"Using device: {self.device}")

    def load_model(self) -> bool:
        """
        Load the embedding model and technique embeddings

        Returns:
            Whether loading was successful
        """
        if self.is_loaded:
            return True

        try:
            # First check if we have a local copy in the models directory
            if os.path.exists(self.models_dir):
                # Check if model files exist
                config_file = os.path.join(self.models_dir, "config.json")
                if os.path.exists(config_file):
                    logger.info(
                        f"Loading semantic model from local cache: {self.models_dir}"
                    )
                    try:
                        # Try loading from cache directory
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.models_dir, use_fast=True, local_files_only=True
                        )

                        self.model = AutoModel.from_pretrained(
                            self.models_dir, local_files_only=True
                        )

                        # Move to device
                        self.model = self.model.to(self.device)

                        # Load technique embeddings
                        if not self._load_technique_embeddings():
                            logger.info("Generating technique embeddings...")
                            self._generate_technique_embeddings()

                        self.is_loaded = True
                        logger.info(
                            f"Semantic model loaded successfully from local cache on {self.device}"
                        )
                        return True
                    except Exception as e:
                        logger.warning(
                            f"Error loading from cache, will try downloading: {str(e)}"
                        )
                        # Continue to download attempt

            # If we reach here, try downloading from Hugging Face
            logger.info(
                f"Downloading semantic model from Hugging Face: BAAI/bge-large-en-v1.5"
            )
            # Set a reasonable timeout for model download
            download_timeout = 60  # seconds

            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "BAAI/bge-large-en-v1.5",
                    cache_dir=self.models_dir,
                    use_fast=True,
                    local_files_only=False,
                )

                # Load model
                self.model = AutoModel.from_pretrained(
                    "BAAI/bge-large-en-v1.5",
                    cache_dir=self.models_dir,
                    local_files_only=False,
                )

                # Move to device
                self.model = self.model.to(self.device)

                # Save model locally for future use
                self.tokenizer.save_pretrained(self.models_dir)
                self.model.save_pretrained(self.models_dir)
                logger.info(f"Saved model to cache directory: {self.models_dir}")

                # Load or generate technique embeddings
                if not self._load_technique_embeddings():
                    logger.info("Generating technique embeddings...")
                    self._generate_technique_embeddings()

                self.is_loaded = True
                logger.info(
                    f"Semantic model downloaded and loaded successfully on {self.device}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to download and load semantic model: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Failed to load semantic model: {str(e)}")
            return False

    def unload_model(self) -> None:
        """Unload the model to free memory"""
        if not self.is_loaded:
            return

        try:
            # Delete model and tokenizer
            del self.model
            del self.tokenizer

            # Set to None
            self.model = None
            self.tokenizer = None

            # Keep embeddings cache in memory if it's not too large

            # Force garbage collection
            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_loaded = False
            logger.info("Semantic model unloaded")

        except Exception as e:
            logger.error(f"Error unloading semantic model: {str(e)}")

    def _load_technique_embeddings(self) -> bool:
        """
        Load cached technique embeddings

        Returns:
            Whether loading was successful
        """
        if not self.cache_embeddings:
            return False

        if not os.path.exists(self.embeddings_cache_path) or not os.path.exists(
            self.embedding_ids_path
        ):
            return False

        try:
            # Load embeddings
            self.technique_embeddings = np.load(self.embeddings_cache_path)

            # Load technique IDs
            with open(self.embedding_ids_path, "r") as f:
                self.embedding_tech_ids = json.load(f)

            logger.info(
                f"Loaded {len(self.embedding_tech_ids)} cached technique embeddings"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load cached embeddings: {str(e)}")
            return False

    def _generate_technique_embeddings(self) -> None:
        """Generate embeddings for all techniques"""
        if not self.is_loaded:
            if not self.load_model():
                logger.error("Model not loaded, cannot generate embeddings")
                return

        try:
            tech_ids = []
            embeddings = []

            # Process techniques in batches
            batch_size = 32
            techniques_list = list(self.techniques.items())

            for i in range(0, len(techniques_list), batch_size):
                batch = techniques_list[i : i + batch_size]
                batch_texts = []
                batch_ids = []

                for tech_id, tech_data in batch:
                    # Skip techniques without keywords
                    if tech_id not in self.technique_keywords:
                        continue

                    # Create rich text representation
                    tech_name = tech_data.get("name", "")
                    tech_desc = tech_data.get("description", "")
                    tech_keywords = self.technique_keywords.get(tech_id, [])

                    # Combine into rich text
                    text = f"{tech_name} - {tech_desc[:500]}"
                    if tech_keywords:
                        text += f" Keywords: {', '.join(tech_keywords)}"

                    batch_texts.append(text)
                    batch_ids.append(tech_id)

                # Get embeddings for batch
                batch_embeddings = self._get_embeddings(batch_texts)

                # Store results
                tech_ids.extend(batch_ids)
                embeddings.extend(batch_embeddings)

                logger.info(
                    f"Generated embeddings for {len(tech_ids)} techniques so far"
                )

            # Convert to numpy array
            self.technique_embeddings = np.array(embeddings)
            self.embedding_tech_ids = tech_ids

            # Cache if enabled
            if self.cache_embeddings:
                np.save(self.embeddings_cache_path, self.technique_embeddings)

                with open(self.embedding_ids_path, "w") as f:
                    json.dump(self.embedding_tech_ids, f)

                logger.info(f"Cached embeddings for {len(tech_ids)} techniques")

        except Exception as e:
            logger.error(f"Error generating technique embeddings: {str(e)}")

    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a list of texts

        Args:
            texts: List of input texts

        Returns:
            List of embeddings
        """
        if not self.is_loaded:
            if not self.load_model():
                logger.error("Model not loaded")
                return [np.zeros(768) for _ in texts]  # Return dummy embeddings

        try:
            embeddings = []

            # Process in smaller batches to avoid OOM
            sub_batch_size = 8

            for i in range(0, len(texts), sub_batch_size):
                sub_batch = texts[i : i + sub_batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    sub_batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)

                    # For BGE models, use mean pooling
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state

                    # Mean pooling
                    input_mask_expanded = (
                        attention_mask.unsqueeze(-1)
                        .expand(token_embeddings.size())
                        .float()
                    )
                    sum_embeddings = torch.sum(
                        token_embeddings * input_mask_expanded, 1
                    )
                    sum_mask = torch.clamp(torch.sum(input_mask_expanded, 1), min=1e-9)
                    mean_embeddings = sum_embeddings / sum_mask

                    # Normalize
                    mean_embeddings = torch.nn.functional.normalize(
                        mean_embeddings, p=2, dim=1
                    )

                    # Convert to numpy and add to results
                    batch_embeddings = mean_embeddings.cpu().numpy()
                    embeddings.extend(batch_embeddings)

            return embeddings

        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            return [np.zeros(768) for _ in texts]  # Return dummy embeddings

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

        return np.dot(a, b) / (norm_a * norm_b)

    # Modify the extract_techniques method in src/extractors/semantic.py
    def extract_techniques(
        self, text: str, threshold: float = 0.5, top_k: int = 10, job_id: str = None
    ) -> List[Dict]:
        """
        Extract techniques using semantic search with metrics recording

        Args:
            text: Input text
            threshold: Minimum similarity threshold
            top_k: Maximum number of results
            job_id: Optional job ID for metrics recording

        Returns:
            List of technique matches with scores
        """
        # Create metrics recorder if job_id is provided
        metrics_recorder = None
        if job_id:
            from src.database.metrics_recorder import MetricsRecorder

            metrics_recorder = MetricsRecorder(job_id)

        # Record start time for performance metrics
        start_time = time.time()

        if self.technique_embeddings is None or len(self.embedding_tech_ids) == 0:
            logger.error("No technique embeddings available")
            return []

        # Record extractor execution details if metrics enabled
        extractor_id = None
        if metrics_recorder:
            parameters = {
                "threshold": threshold,
                "top_k": top_k,
                "model_name": self.model_name,
                "device": self.device,
            }
            extractor_id = metrics_recorder.record_extractor_result(
                extractor_name="semantic",
                raw_input=text,
                raw_output={},  # Will update this later
                execution_time_ms=0,  # Will update this later
                parameters=parameters,
            )

            # Record model details
            if self.is_loaded:
                metrics_recorder.record_model_weights(
                    extractor_id=extractor_id,
                    model_name=self.model_name,
                    weight_path=self.models_dir,
                    weight_size_bytes=None,  # Would require additional code to calculate
                    quantization_bits=16
                    if self.device == "cuda"
                    else 32,  # Typical values
                )

            # Record performance benchmark for model loading
            load_time_start = time.time()
            if not self.is_loaded:
                self.load_model()
            load_time_ms = int((time.time() - load_time_start) * 1000)

            if load_time_ms > 0:  # Only record if we actually loaded the model
                metrics_recorder.record_performance_benchmark(
                    extractor_name="semantic",
                    operation_type="model_loading",
                    execution_time_ms=load_time_ms,
                )

        try:
            # Record embedding computation
            embed_start_time = time.time()
            query_embeddings = self._get_embeddings([text])
            embed_time_ms = int((time.time() - embed_start_time) * 1000)

            if not query_embeddings:
                logger.error("Failed to get query embedding")
                return []

            query_embedding = query_embeddings[0]

            # Record embedding details
            if metrics_recorder and extractor_id:
                # Record text segment first
                segment_id = metrics_recorder.record_text_segment(
                    text=text, index=0, embedding=query_embedding
                )

                # Record embedding details
                metrics_recorder.record_embedding_details(
                    extractor_id=extractor_id,
                    text_segment_id=segment_id,
                    embedding_type="query",
                    embedding_model=self.model_name,
                    embedding_dimension=len(query_embedding),
                    normalization_applied=True,  # BGE models typically normalize
                    cache_hit=False,
                    approximate_search_used=False,
                )

                # Record embedding performance
                metrics_recorder.record_performance_benchmark(
                    extractor_name="semantic",
                    operation_type="text_embedding",
                    execution_time_ms=embed_time_ms,
                    input_size=len(text),
                    throughput_tokens_per_second=len(text) / (embed_time_ms / 1000)
                    if embed_time_ms > 0
                    else None,
                )

            # Calculate similarities
            similarities = []
            similarity_start_time = time.time()

            for i, tech_id in enumerate(self.embedding_tech_ids):
                if i >= len(self.technique_embeddings):
                    continue

                tech_embedding = self.technique_embeddings[i]
                similarity = self._cosine_similarity(query_embedding, tech_embedding)

                similarities.append((tech_id, similarity))

                # Record technique embedding details if metrics enabled
                if metrics_recorder and extractor_id and similarity >= threshold:
                    metrics_recorder.record_embedding_details(
                        extractor_id=extractor_id,
                        technique_id=tech_id,
                        embedding_type="technique",
                        embedding_model=self.model_name,
                        embedding_dimension=len(tech_embedding),
                        normalization_applied=True,
                        cosine_similarity=similarity,
                    )

            similarity_time_ms = int((time.time() - similarity_start_time) * 1000)

            # Record similarity computation performance
            if metrics_recorder and extractor_id:
                metrics_recorder.record_performance_benchmark(
                    extractor_name="semantic",
                    operation_type="similarity_computation",
                    execution_time_ms=similarity_time_ms,
                    input_size=len(self.embedding_tech_ids),
                )

            # Filter by threshold
            filtered_similarities = [
                (tech_id, sim) for tech_id, sim in similarities if sim >= threshold
            ]

            # Sort by similarity and limit results
            filtered_similarities.sort(key=lambda x: x[1], reverse=True)
            filtered_similarities = filtered_similarities[:top_k]

            # Convert to result format
            results = []

            for tech_id, similarity in filtered_similarities:
                result = {
                    "technique_id": tech_id,
                    "confidence": similarity,
                    "method": "semantic",
                }

                # Record semantic scores if metrics enabled
                if metrics_recorder and extractor_id:
                    metrics_recorder.record_semantic_scores(
                        extractor_id=extractor_id,
                        technique_id=tech_id,
                        similarity_score=similarity,
                        embedding_dimension=len(query_embedding),
                        model_used=self.model_name,
                    )

                results.append(result)

            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Update extractor result with final output and execution time
            if metrics_recorder and extractor_id:
                # Update the extractor result with the actual output and execution time
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
            logger.error(f"Error extracting techniques with semantic search: {str(e)}")

            # Record error if metrics enabled
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
