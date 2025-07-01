# src/database/metrics_recorder.py

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from psycopg2.extras import Json

from src.database.postgresql import get_db

logger = logging.getLogger(__name__)


class MetricsRecorder:
    """Comprehensive recorder for model metrics and artifacts"""

    def __init__(self, job_id: str):
        """
        Initialize the metrics recorder

        Args:
            job_id: ID of the analysis job
        """
        self.job_id = job_id
        self.db = get_db()

    def record_text_segment(
        self, text: str, index: int, embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Record a text segment and its embedding

        Args:
            text: Segment text
            index: Segment index for ordering
            embedding: Optional vector embedding

        Returns:
            ID of the created record
        """
        segment_id = str(uuid.uuid4())

        # Convert embedding to bytes if provided
        embedding_bytes = None
        if embedding is not None:
            embedding_bytes = embedding.tobytes()

        query = """
        INSERT INTO analysis_text_segments
        (id, job_id, segment_text, segment_index, segment_embedding)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """

        result = self.db.query_one(
            query, (segment_id, self.job_id, text, index, embedding_bytes)
        )
        return str(result["id"])

    def record_extractor_result(
        self,
        extractor_name: str,
        raw_input: str,
        raw_output: Dict,
        execution_time_ms: int,
        parameters: Dict,
    ) -> str:
        """
        Record detailed extractor results

        Args:
            extractor_name: Name of the extractor
            raw_input: Input sent to the extractor
            raw_output: Full output from the extractor
            execution_time_ms: Execution time in milliseconds
            parameters: Parameters used for the extractor

        Returns:
            ID of the created record
        """
        extractor_id = str(uuid.uuid4())

        query = """
        INSERT INTO extractor_results
        (id, job_id, extractor_name, raw_input, raw_output, execution_time_ms, parameters)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        result = self.db.query_one(
            query,
            (
                extractor_id,
                self.job_id,
                extractor_name,
                raw_input,
                Json(raw_output),
                execution_time_ms,
                Json(parameters),
            ),
        )

        return str(result["id"])

    def record_entities(self, extractor_id: str, entities: List[Dict]) -> None:
        """
        Record entities identified by NER

        Args:
            extractor_id: ID of the extractor result
            entities: List of entity dictionaries
        """
        if not entities:
            return

        for entity in entities:
            entity_id = str(uuid.uuid4())

            query = """
            INSERT INTO analysis_entities
            (id, job_id, extractor_id, entity_text, entity_type, start_offset, end_offset, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            self.db.execute(
                query,
                (
                    entity_id,
                    self.job_id,
                    extractor_id,
                    entity.get("text", ""),
                    entity.get("type", "UNKNOWN"),
                    entity.get("start_offset"),
                    entity.get("end_offset"),
                    entity.get("confidence", 0.0),
                ),
            )

    def record_keywords(
        self, extractor_id: str, technique_id: str, keywords: List[Dict]
    ) -> None:
        """
        Record matched keywords

        Args:
            extractor_id: ID of the extractor result
            technique_id: ID of the ATT&CK technique
            keywords: List of keyword dictionaries
        """
        if not keywords:
            return

        for keyword in keywords:
            keyword_id = str(uuid.uuid4())

            query = """
            INSERT INTO analysis_keywords
            (id, job_id, extractor_id, technique_id, keyword, match_position, match_context)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            self.db.execute(
                query,
                (
                    keyword_id,
                    self.job_id,
                    extractor_id,
                    technique_id,
                    keyword.get("text", ""),
                    keyword.get("position"),
                    keyword.get("context", ""),
                ),
            )

    def record_bm25_scores(
        self,
        extractor_id: str,
        technique_id: str,
        raw_score: float,
        normalized_score: float,
        matched_terms: Dict,
    ) -> None:
        """
        Record BM25 scores

        Args:
            extractor_id: ID of the extractor result
            technique_id: ID of the ATT&CK technique
            raw_score: Raw BM25 score
            normalized_score: Normalized score (0-1)
            matched_terms: Terms that matched
        """
        query = """
        INSERT INTO bm25_scores
        (id, job_id, extractor_id, technique_id, raw_score, normalized_score, matched_terms)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        score_id = str(uuid.uuid4())
        self.db.execute(
            query,
            (
                score_id,
                self.job_id,
                extractor_id,
                technique_id,
                raw_score,
                normalized_score,
                Json(matched_terms),
            ),
        )

    def record_semantic_scores(
        self,
        extractor_id: str,
        technique_id: str,
        similarity_score: float,
        embedding_dimension: int,
        model_used: str,
    ) -> None:
        """
        Record semantic similarity scores

        Args:
            extractor_id: ID of the extractor result
            technique_id: ID of the ATT&CK technique
            similarity_score: Similarity score (0-1)
            embedding_dimension: Dimension of embeddings used
            model_used: Name of the model used
        """
        query = """
        INSERT INTO semantic_scores
        (id, job_id, extractor_id, technique_id, similarity_score, embedding_dimension, model_used)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        score_id = str(uuid.uuid4())
        self.db.execute(
            query,
            (
                score_id,
                self.job_id,
                extractor_id,
                technique_id,
                similarity_score,
                embedding_dimension,
                model_used,
            ),
        )

    def record_ensemble_details(
        self,
        technique_id: str,
        ensemble_method: str,
        final_confidence: float,
        component_scores: Dict,
        weights_used: Dict,
    ) -> None:
        """
        Record ensemble method details

        Args:
            technique_id: ID of the ATT&CK technique
            ensemble_method: Name of the ensemble method used
            final_confidence: Final confidence score
            component_scores: Scores from each component extractor
            weights_used: Weights used in ensemble
        """
        query = """
        INSERT INTO ensemble_details
        (id, job_id, technique_id, ensemble_method, final_confidence, component_scores, weights_used)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        ensemble_id = str(uuid.uuid4())
        self.db.execute(
            query,
            (
                ensemble_id,
                self.job_id,
                technique_id,
                ensemble_method,
                final_confidence,
                Json(component_scores),
                Json(weights_used),
            ),
        )

    def record_model_metrics(
        self,
        extractor_id: str,
        model_name: str,
        model_version: str = None,
        device_used: str = None,
        memory_usage_mb: float = None,
        batch_size: int = None,
        quantization_used: bool = None,
        execution_time_ms: int = None,
    ) -> None:
        """
        Record model metrics and parameters

        Args:
            extractor_id: ID of the extractor result
            model_name: Name of the model
            model_version: Version of the model
            device_used: Device used for inference
            memory_usage_mb: Memory usage in MB
            batch_size: Batch size used
            quantization_used: Whether quantization was used
            execution_time_ms: Execution time in milliseconds
        """
        query = """
        INSERT INTO model_execution_metrics
        (id, job_id, extractor_id, model_name, model_version, device_used, 
         memory_usage_mb, batch_size, quantization_used, execution_time_ms)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        metrics_id = str(uuid.uuid4())
        self.db.execute(
            query,
            (
                metrics_id,
                self.job_id,
                extractor_id,
                model_name,
                model_version,
                device_used,
                memory_usage_mb,
                batch_size,
                quantization_used,
                execution_time_ms,
            ),
        )

    # Methods for specific extractors

    def record_ner_details(
        self,
        extractor_id: str,
        entity_count: int,
        entity_types: Dict[str, int],
        model_name: str,
        aggregation_strategy: str = None,
        tokenizer_max_length: int = None,
    ) -> str:
        """
        Record NER-specific details

        Args:
            extractor_id: ID of the extractor result
            entity_count: Total number of entities found
            entity_types: Count of each entity type
            model_name: Name of the NER model
            aggregation_strategy: Strategy used for aggregation
            tokenizer_max_length: Maximum token length

        Returns:
            ID of the created record
        """
        detail_id = str(uuid.uuid4())

        query = """
        INSERT INTO ner_extraction_details
        (id, job_id, extractor_id, entity_count, entity_types, model_name, aggregation_strategy, tokenizer_max_length)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        result = self.db.query_one(
            query,
            (
                detail_id,
                self.job_id,
                extractor_id,
                entity_count,
                Json(entity_types),
                model_name,
                aggregation_strategy,
                tokenizer_max_length,
            ),
        )

        return str(result["id"])

    def record_embedding_details(
        self,
        extractor_id: str,
        text_segment_id: str = None,
        technique_id: str = None,
        embedding_type: str = None,
        embedding_model: str = None,
        embedding_dimension: int = None,
        normalization_applied: bool = None,
        cache_hit: bool = None,
        cosine_similarity: float = None,
        approximate_search_used: bool = None,
    ) -> str:
        """
        Record embedding computation details

        Args:
            extractor_id: ID of the extractor result
            text_segment_id: ID of the text segment
            technique_id: ID of the ATT&CK technique
            embedding_type: Type of embedding
            embedding_model: Model used for embedding
            embedding_dimension: Dimension of the embedding
            normalization_applied: Whether normalization was applied
            cache_hit: Whether the embedding was found in cache
            cosine_similarity: Cosine similarity score if applicable
            approximate_search_used: Whether approximate search was used

        Returns:
            ID of the created record
        """
        detail_id = str(uuid.uuid4())

        query = """
        INSERT INTO embedding_details
        (id, job_id, extractor_id, text_segment_id, technique_id, embedding_type, 
         embedding_model, embedding_dimension, normalization_applied, cache_hit, 
         cosine_similarity, approximate_search_used)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        result = self.db.query_one(
            query,
            (
                detail_id,
                self.job_id,
                extractor_id,
                text_segment_id,
                technique_id,
                embedding_type,
                embedding_model,
                embedding_dimension,
                normalization_applied,
                cache_hit,
                cosine_similarity,
                approximate_search_used,
            ),
        )

        return str(result["id"])

    def record_kev_details(
        self,
        extractor_id: str,
        cve_id: str,
        cve_mention_position: int = None,
        cve_mention_context: str = None,
        kev_entry_date: datetime = None,
        technique_mappings: Dict = None,
        confidence_scores: Dict = None,
    ) -> str:
        """
        Record KEV extraction details

        Args:
            extractor_id: ID of the extractor result
            cve_id: CVE identifier
            cve_mention_position: Position of CVE mention in text
            cve_mention_context: Context around CVE mention
            kev_entry_date: Date the CVE was added to KEV catalog
            technique_mappings: Techniques mapped to this CVE
            confidence_scores: Confidence scores for mappings

        Returns:
            ID of the created record
        """
        detail_id = str(uuid.uuid4())

        query = """
        INSERT INTO kev_extraction_details
        (id, job_id, extractor_id, cve_id, cve_mention_position, cve_mention_context,
         kev_entry_date, technique_mappings, confidence_scores)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        result = self.db.query_one(
            query,
            (
                detail_id,
                self.job_id,
                extractor_id,
                cve_id,
                cve_mention_position,
                cve_mention_context,
                kev_entry_date,
                Json(technique_mappings) if technique_mappings else None,
                Json(confidence_scores) if confidence_scores else None,
            ),
        )

        return str(result["id"])

    def record_classifier_details(
        self,
        extractor_id: str,
        model_type: str,
        feature_count: int = None,
        class_count: int = None,
        probability_scores: Dict = None,
        decision_threshold: float = None,
        embedding_used: bool = None,
    ) -> str:
        """
        Record classifier-specific details

        Args:
            extractor_id: ID of the extractor result
            model_type: Type of classifier model
            feature_count: Number of features used
            class_count: Number of classes
            probability_scores: Raw probability scores
            decision_threshold: Decision threshold used
            embedding_used: Whether embeddings were used

        Returns:
            ID of the created record
        """
        detail_id = str(uuid.uuid4())

        query = """
        INSERT INTO classifier_details
        (id, job_id, extractor_id, model_type, feature_count, class_count,
         probability_scores, decision_threshold, embedding_used)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        result = self.db.query_one(
            query,
            (
                detail_id,
                self.job_id,
                extractor_id,
                model_type,
                feature_count,
                class_count,
                Json(probability_scores) if probability_scores else None,
                decision_threshold,
                embedding_used,
            ),
        )

        return str(result["id"])

    def record_model_weights(
        self,
        extractor_id: str,
        model_name: str,
        weight_path: str = None,
        weight_hash: str = None,
        weight_size_bytes: int = None,
        quantization_bits: int = None,
    ) -> str:
        """
        Record model weight details

        Args:
            extractor_id: ID of the extractor result
            model_name: Name of the model
            weight_path: Path to model weights
            weight_hash: Hash of model weights
            weight_size_bytes: Size of weights in bytes
            quantization_bits: Bit precision of weights

        Returns:
            ID of the created record
        """
        weight_id = str(uuid.uuid4())

        query = """
        INSERT INTO model_weights
        (id, job_id, extractor_id, model_name, weight_path, weight_hash,
         weight_size_bytes, quantization_bits)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        result = self.db.query_one(
            query,
            (
                weight_id,
                self.job_id,
                extractor_id,
                model_name,
                weight_path,
                weight_hash,
                weight_size_bytes,
                quantization_bits,
            ),
        )

        return str(result["id"])

    def record_performance_benchmark(
        self,
        extractor_name: str,
        operation_type: str,
        execution_time_ms: int,
        input_size: int = None,
        throughput_tokens_per_second: float = None,
        memory_peak_mb: float = None,
    ) -> str:
        """
        Record performance benchmark

        Args:
            extractor_name: Name of the extractor
            operation_type: Type of operation
            execution_time_ms: Execution time in milliseconds
            input_size: Size of input
            throughput_tokens_per_second: Throughput in tokens per second
            memory_peak_mb: Peak memory usage in MB

        Returns:
            ID of the created record
        """
        benchmark_id = str(uuid.uuid4())

        query = """
        INSERT INTO performance_benchmarks
        (id, job_id, extractor_name, operation_type, input_size, execution_time_ms,
         throughput_tokens_per_second, memory_peak_mb)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        result = self.db.query_one(
            query,
            (
                benchmark_id,
                self.job_id,
                extractor_name,
                operation_type,
                input_size,
                execution_time_ms,
                throughput_tokens_per_second,
                memory_peak_mb,
            ),
        )

        return str(result["id"])


# Add a function to retrieve all metrics for a job
def get_complete_job_metrics(job_id: str) -> Dict:
    """
    Get complete metrics for a job

    Args:
        job_id: ID of the analysis job

    Returns:
        Dictionary with all job metrics
    """
    db = get_db()

    # Get job metadata
    job = db.query_one(
        """
        SELECT * FROM analysis_jobs WHERE id = %s
        """,
        (job_id,),
    )

    if not job:
        return {"error": "Job not found"}

    # Get all related data
    results = {}

    # Get preprocessor details
    results["preprocessing"] = db.query_one(
        "SELECT * FROM preprocessing_details WHERE job_id = %s", (job_id,)
    )

    # Get text segments
    results["segments"] = db.query(
        "SELECT id, segment_text, segment_index FROM analysis_text_segments WHERE job_id = %s",
        (job_id,),
    )

    # Get extractor results
    extractors = db.query(
        "SELECT * FROM extractor_results WHERE job_id = %s", (job_id,)
    )

    # Organize results by extractor
    results["extractors"] = {}

    for extractor in extractors:
        extractor_id = extractor["id"]
        extractor_name = extractor["extractor_name"]

        # Get all related data for this extractor
        extractor_data = dict(extractor)

        # Add entities
        extractor_data["entities"] = db.query(
            "SELECT * FROM analysis_entities WHERE extractor_id = %s", (extractor_id,)
        )

        # Add keywords
        extractor_data["keywords"] = db.query(
            "SELECT * FROM analysis_keywords WHERE extractor_id = %s", (extractor_id,)
        )

        # Add BM25 scores
        extractor_data["bm25_scores"] = db.query(
            "SELECT * FROM bm25_scores WHERE extractor_id = %s", (extractor_id,)
        )

        # Add semantic scores
        extractor_data["semantic_scores"] = db.query(
            "SELECT * FROM semantic_scores WHERE extractor_id = %s", (extractor_id,)
        )

        # Add model metrics
        extractor_data["model_metrics"] = db.query(
            "SELECT * FROM model_execution_metrics WHERE extractor_id = %s",
            (extractor_id,),
        )

        results["extractors"][extractor_name] = extractor_data

    # Get ensemble details
    results["ensemble_details"] = db.query(
        "SELECT * FROM ensemble_details WHERE job_id = %s", (job_id,)
    )

    # Get technique relationships
    results["relationships"] = db.query(
        "SELECT * FROM technique_relationships WHERE job_id = %s", (job_id,)
    )

    # Get final analysis results
    results["analysis_results"] = db.query(
        "SELECT * FROM analysis_results WHERE job_id = %s", (job_id,)
    )

    return results
