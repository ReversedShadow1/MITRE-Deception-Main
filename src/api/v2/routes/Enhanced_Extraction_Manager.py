"""
Enhanced Extraction Manager for V2 API
------------------------------------
Coordinates multiple extractors with optimizations, caching, and asynchronous processing.
"""

import asyncio
import concurrent.futures
import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.api.v2.routes.Advanced_Ensemble import AdvancedEnsembleMethod
from src.api.v2.routes.bm25_extractor_V2 import EnhancedBM25Extractor

# Import caching layer
from src.api.v2.routes.Extraction_Result_Cache import (
    ExtractionResultCache,
    RequestLimiter,
)
from src.api.v2.routes.Optimized_Neural_Extractor import (
    OptimizedEmbeddingExtractor,
    OptimizedNERExtractor,
)

# Import optimized extractors
from src.api.v2.routes.rule_based_V2 import EnhancedRuleBasedExtractor

# Import for Neo4j integration
from src.database.neo4j import get_neo4j

# Import advanced ensemble
from src.database.postgresql import get_db
from src.extractors.bm25_extractor import BM25Extractor
from src.extractors.kev_extractor import KEVExtractor
from src.extractors.ner_extractor import SecureBERTNERExtractor

# Import extractors
from src.extractors.rule_based import RuleBasedExtractor
from src.extractors.semantic import BGEEmbeddingExtractor

logger = logging.getLogger("EnhancedExtractionManager")


class EnhancedExtractionManager:
    """
    Manager for enhanced extraction operations with optimizations, caching, and
    asynchronous processing capabilities
    """

    def __init__(
        self,
        techniques_data: Dict,
        technique_keywords: Dict,
        use_optimized_extractors: bool = True,
        use_caching: bool = True,
        use_async: bool = True,
        cache_type: str = "memory",
        cache_dir: str = "cache/extraction",
        redis_url: str = None,
        use_neo4j: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize enhanced extraction manager

        Args:
            techniques_data: Dictionary of technique data
            technique_keywords: Dictionary of technique keywords
            use_optimized_extractors: Whether to use optimized extractors
            use_caching: Whether to use result caching
            use_async: Whether to use asynchronous processing
            cache_type: Type of cache ('memory', 'file', or 'redis')
            cache_dir: Directory for file cache
            redis_url: URL for Redis connection
            use_neo4j: Whether to use Neo4j for relationship-based boosting
            max_workers: Maximum number of concurrent workers
        """
        self.techniques_data = techniques_data
        self.technique_keywords = technique_keywords
        self.use_optimized_extractors = use_optimized_extractors
        self.use_caching = use_caching
        self.use_async = use_async
        self.use_neo4j = use_neo4j
        self.max_workers = max_workers

        from src.database.postgresql import get_db

        self.db = get_db()

        # Connect to Neo4j if enabled
        self.neo4j = None
        if self.use_neo4j:
            try:
                self.neo4j = get_neo4j()
                logger.info("Connected to Neo4j")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                self.use_neo4j = False

        # Initialize cache if enabled
        self.cache = None
        if self.use_caching:
            self.cache = ExtractionResultCache(
                cache_type=cache_type,
                cache_dir=cache_dir,
                redis_url=redis_url,
                ttl=86400,  # 24 hours TTL
            )
            logger.info(f"Initialized result cache ({cache_type})")

        # Initialize request limiter
        self.limiter = RequestLimiter(redis_url=redis_url)

        # Initialize extractors
        self.extractors = {}
        self._initialize_extractors()

        # Initialize executor for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Initialize advanced ensemble
        self.ensemble = AdvancedEnsembleMethod(
            techniques_data=techniques_data,
            technique_keywords=technique_keywords,
            use_neo4j=use_neo4j,
            neo4j_connector=self.neo4j,
        )

        logger.info("Enhanced extraction manager initialized")

    def _initialize_extractors(self) -> None:
        """Initialize all extractors"""
        # Use optimized extractors if enabled
        if self.use_optimized_extractors:
            self._initialize_optimized_extractors()
        else:
            self._initialize_original_extractors()

    def _initialize_optimized_extractors(self) -> None:
        """Initialize optimized extractors"""
        # Enhanced rule-based extractor
        self.extractors["rule_based"] = EnhancedRuleBasedExtractor(
            technique_keywords=self.technique_keywords,
            techniques_data=self.techniques_data,
            neo4j_connector=self.neo4j,
            use_aho_corasick=True,
            use_contextual_boost=True,
        )

        # Enhanced BM25 extractor
        self.extractors["bm25"] = EnhancedBM25Extractor(
            techniques=self.techniques_data,
            technique_keywords=self.technique_keywords,
            models_dir="models/enhanced_bm25",
            bm25_variant="plus",
            use_field_weighting=True,
            neo4j_connector=self.neo4j,
        )

        # Optimized NER extractor
        self.extractors["ner"] = OptimizedNERExtractor(
            techniques_data=self.techniques_data,
            technique_keywords=self.technique_keywords,
            model_name="CyberPeace-Institute/SecureBERT-NER",
            cache_dir="models/optimized_ner",
            use_gpu=True,
            use_quantization=True,
            batch_size=16,
            use_model_cache=True,
            confidence_calibration=True,
            neo4j_connector=self.neo4j,
        )

        # Optimized semantic extractor
        self.extractors["semantic"] = OptimizedEmbeddingExtractor(
            techniques_data=self.techniques_data,
            technique_keywords=self.technique_keywords,
            model_name="BAAI/bge-large-en-v1.5",
            cache_dir="models/optimized_embedding",
            embeddings_dir="models/optimized_embeddings",
            use_gpu=True,
            use_quantization=True,
            batch_size=16,
            use_model_cache=True,
            cache_embeddings=True,
            use_approximate_search=True,
            neo4j_connector=self.neo4j,
        )

        # KEV extractor (using original implementation)
        try:
            from src.integrations.kev_mapper import KEVMapper

            kev_mapper = KEVMapper(data_dir="data")
            kev_mapper.load_kev_data()
            kev_mapper.load_cve_attack_mappings()

            self.extractors["kev"] = KEVExtractor(
                kev_mapper=kev_mapper,
                techniques_data=self.techniques_data,
                neo4j_connector=self.neo4j,
            )
        except Exception as e:
            logger.error(f"Failed to initialize KEV extractor: {e}")
            self.extractors["kev"] = None

    def _initialize_original_extractors(self) -> None:
        """Initialize original extractors for backward compatibility"""
        # Rule-based extractor
        self.extractors["rule_based"] = RuleBasedExtractor(
            technique_keywords=self.technique_keywords,
            techniques_data=self.techniques_data,
            neo4j_connector=self.neo4j,
        )

        # BM25 extractor
        self.extractors["bm25"] = BM25Extractor(
            techniques=self.techniques_data,
            technique_keywords=self.technique_keywords,
            models_dir="models/bm25",
            bm25_variant="plus",
            neo4j_connector=self.neo4j,
        )

        # NER extractor
        self.extractors["ner"] = SecureBERTNERExtractor(
            model_name="CyberPeace-Institute/SecureBERT-NER",
            techniques_data=self.techniques_data,
            technique_keywords=self.technique_keywords,
            cache_dir="models/SecureBERT-NER",
            use_gpu=True,
            neo4j_connector=self.neo4j,
        )

        # Semantic extractor
        self.extractors["semantic"] = BGEEmbeddingExtractor(
            model_name="BAAI/bge-large-en-v1.5",
            techniques=self.techniques_data,
            technique_keywords=self.technique_keywords,
            models_dir="models/bge-large-en-v1.5",
            use_gpu=True,
            neo4j_connector=self.neo4j,
        )

        # KEV extractor
        try:
            from src.integrations.kev_mapper import KEVMapper

            kev_mapper = KEVMapper(data_dir="data")
            kev_mapper.load_kev_data()
            kev_mapper.load_cve_attack_mappings()

            self.extractors["kev"] = KEVExtractor(
                kev_mapper=kev_mapper,
                techniques_data=self.techniques_data,
                neo4j_connector=self.neo4j,
            )
        except Exception as e:
            logger.error(f"Failed to initialize KEV extractor: {e}")
            self.extractors["kev"] = None

    def check_request_limits(
        self, user_id: str, tier: str, text_length: int, batch_size: int = 1
    ) -> Tuple[bool, Dict]:
        """
        Check if request is within limits

        Args:
            user_id: User identifier
            tier: User tier
            text_length: Length of text in request
            batch_size: Size of batch in request

        Returns:
            Tuple of (is_allowed, limit_info)
        """
        return self.limiter.check_limit(
            user_id=user_id, tier=tier, text_length=text_length, batch_size=batch_size
        )

    # Modify in src/api/v2/routes/Enhanced_Extraction_Manager.py

    def extract_techniques(
        self,
        text: str,
        extractors: List[str] = None,
        threshold: float = 0.2,
        top_k: int = 10,
        use_ensemble: bool = True,
        include_context: bool = False,
        include_relationships: bool = False,
        return_navigator_layer: bool = False,
        user_id: str = None,
        tier: str = "basic",
        request_id: str = None,
    ) -> Dict:
        """
        Extract ATT&CK techniques with enhanced capabilities and metric recording

        Args:
            text: Input text
            extractors: List of extractors to use
            threshold: Minimum confidence threshold
            top_k: Maximum number of results
            use_ensemble: Whether to use ensemble method
            include_context: Whether to include contextual information
            include_relationships: Whether to include technique relationships
            return_navigator_layer: Whether to return MITRE Navigator layer
            user_id: User identifier for caching and rate limiting
            tier: User tier for rate limiting
            request_id: Request identifier for tracking

        Returns:
            Dictionary with extraction results
        """
        # Check if input is valid
        if not text or not text.strip():
            return {
                "techniques": [],
                "meta": {
                    "text_length": 0,
                    "processing_time": 0,
                    "error": "Empty text provided",
                },
            }

        # Generate job ID for tracking
        import uuid
        from datetime import datetime

        job_id = str(uuid.uuid4())

        # Create metrics recorder
        from src.database.metrics_recorder import MetricsRecorder

        metrics_recorder = MetricsRecorder(job_id)

        # Use default extractors if none specified
        if not extractors:
            extractors = ["rule_based", "bm25", "ner", "kev"]

        # Check if we have a cached result
        if self.use_caching and self.cache:
            cached_result = self.cache.get(
                text=text,
                extractors=extractors,
                threshold=threshold,
                top_k=top_k,
                use_ensemble=use_ensemble,
            )

            if cached_result:
                logger.info(f"Retrieved result from cache (request_id={request_id})")

                # Add request-specific metadata
                cached_result["meta"]["from_cache"] = True
                cached_result["meta"]["job_id"] = job_id

                # Record that a cached result was used
                self.db.execute(
                    """
                    INSERT INTO analysis_jobs
                    (id, user_id, status, input_type, input_data, extractors_used, threshold, completed_at, processing_time_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job_id,
                        user_id or "anonymous",
                        "completed_from_cache",
                        "text",
                        text[:1000],  # Store first 1000 chars
                        extractors,
                        threshold,
                        datetime.now(),
                        0,  # No processing time for cached results
                    ),
                )

                return cached_result

        # Start timing
        start_time = time.time()

        # Record the job start
        self.db.execute(
            """
            INSERT INTO analysis_jobs
            (id, user_id, status, input_type, input_data, extractors_used, threshold)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                job_id,
                user_id or "anonymous",
                "processing",
                "text",
                text[:1000],  # Store first 1000 chars
                extractors,
                threshold,
            ),
        )

        # Extract techniques
        try:
            if self.use_async:
                # Use asynchronous extraction
                result = self._extract_async(
                    text=text,
                    extractors=extractors,
                    threshold=threshold,
                    top_k=top_k,
                    use_ensemble=use_ensemble,
                    job_id=job_id,  # Pass job_id for metrics recording
                )
            else:
                # Use synchronous extraction
                result = self._extract_sync(
                    text=text,
                    extractors=extractors,
                    threshold=threshold,
                    top_k=top_k,
                    use_ensemble=use_ensemble,
                    job_id=job_id,  # Pass job_id for metrics recording
                )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Build response
            techniques = result.get("techniques", [])

            response = {
                "techniques": techniques,
                "meta": {
                    "text_length": len(text),
                    "processing_time": round(processing_time, 3),
                    "extractors_used": {extractor: True for extractor in extractors},
                    "ensemble_used": use_ensemble,
                    "threshold": threshold,
                    "technique_count": len(techniques),
                    "using_neo4j": self.use_neo4j,
                    "request_id": request_id,
                    "job_id": job_id,
                    "from_cache": False,
                },
            }

            # Add context if requested
            if include_context:
                self._add_context(response, techniques)

            # Add relationships if requested
            if include_relationships:
                self._add_relationships(response, techniques)

            # Add Navigator layer if requested
            if return_navigator_layer:
                response["navigator_layer"] = self._generate_navigator_layer(techniques)

            # Update the job record with completion status
            self.db.execute(
                """
                UPDATE analysis_jobs
                SET status = %s, completed_at = %s, processing_time_ms = %s
                WHERE id = %s
                """,
                ("completed", datetime.now(), int(processing_time * 1000), job_id),
            )

            # Store analysis results in database
            for technique in techniques:
                result_id = str(uuid.uuid4())
                self.db.execute(
                    """
                    INSERT INTO analysis_results
                    (id, job_id, technique_id, technique_name, confidence, method, matched_keywords, cve_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        result_id,
                        job_id,
                        technique.get("technique_id"),
                        technique.get("name"),
                        technique.get("confidence"),
                        technique.get("method"),
                        technique.get("matched_keywords"),
                        technique.get("cve_id"),
                    ),
                )

            # Cache result if enabled
            if self.use_caching and self.cache:
                self.cache.set(
                    text=text,
                    result=response,
                    extractors=extractors,
                    threshold=threshold,
                    top_k=top_k,
                    use_ensemble=use_ensemble,
                )

            return response

        except Exception as e:
            # Log error
            logger.error(f"Error extracting techniques: {e}", exc_info=True)

            # Update job with error status
            self.db.execute(
                """
                UPDATE analysis_jobs
                SET status = %s, completed_at = %s, processing_time_ms = %s
                WHERE id = %s
                """,
                (
                    "failed",
                    datetime.now(),
                    int((time.time() - start_time) * 1000),
                    job_id,
                ),
            )

            # Return error response
            return {
                "techniques": [],
                "meta": {
                    "text_length": len(text),
                    "processing_time": time.time() - start_time,
                    "error": str(e),
                    "request_id": request_id,
                    "job_id": job_id,
                },
            }

    def _extract_sync(
        self,
        text: str,
        extractors: List[str],
        threshold: float,
        top_k: int,
        use_ensemble: bool,
        job_id: str = None,  # Add job_id parameter
    ) -> Dict:
        """
        Extract techniques synchronously with metrics recording

        Args:
            text: Input text
            extractors: List of extractors to use
            threshold: Minimum confidence threshold
            top_k: Maximum number of results
            use_ensemble: Whether to use ensemble method
            job_id: Optional job ID for metrics recording

        Returns:
            Dictionary with extraction results
        """
        # Store results from each extractor
        extractor_results = {}

        # Process each extractor
        for extractor_name in extractors:
            # Skip if extractor not available
            if (
                extractor_name not in self.extractors
                or self.extractors[extractor_name] is None
            ):
                logger.warning(f"Extractor {extractor_name} not available")
                continue

            # Get extractor
            extractor = self.extractors[extractor_name]

            # Extract techniques
            logger.info(f"Extracting techniques with {extractor_name}")

            # Handle different parameter naming conventions
            if extractor_name in [
                "rule_based",
                "ner",
                "enhanced_rule_based",
                "optimized_ner",
            ]:
                # These extractors expect min_confidence and max_results
                results = extractor.extract_techniques(
                    text=text,
                    min_confidence=threshold,
                    max_results=top_k,
                    job_id=job_id,  # Pass job_id for metrics recording
                )
            else:
                # Other extractors expect threshold and top_k
                results = extractor.extract_techniques(
                    text=text,
                    threshold=threshold,
                    top_k=top_k,
                    job_id=job_id,  # Pass job_id for metrics recording
                )

            # Store results
            extractor_results[extractor_name] = results

        # Use ensemble if requested
        if use_ensemble and extractor_results:
            # Use advanced ensemble
            techniques = self.ensemble.ensemble_extractors(
                text=text,
                extractor_results=extractor_results,
                threshold=threshold,
                max_results=top_k,
                job_id=job_id,  # Pass job_id for metrics recording
            )
        else:
            # Combine results without ensemble
            all_results = []
            for results in extractor_results.values():
                all_results.extend(results)

            # Remove duplicates, sort by confidence, and limit results
            seen_techniques = set()
            techniques = []

            for result in sorted(
                all_results, key=lambda x: x.get("confidence", 0), reverse=True
            ):
                tech_id = result.get("technique_id")
                if tech_id and tech_id not in seen_techniques:
                    seen_techniques.add(tech_id)
                    techniques.append(result)

                    if len(techniques) >= top_k:
                        break

        return {"techniques": techniques}

    def _extract_async(
        self,
        text: str,
        extractors: List[str],
        threshold: float,
        top_k: int,
        use_ensemble: bool,
        job_id: str = None,  # Add job_id parameter
    ) -> Dict:
        """
        Extract techniques asynchronously with metrics recording

        Args:
            text: Input text
            extractors: List of extractors to use
            threshold: Minimum confidence threshold
            top_k: Maximum number of results
            use_ensemble: Whether to use ensemble method
            job_id: Optional job ID for metrics recording

        Returns:
            Dictionary with extraction results
        """
        # Store results from each extractor
        extractor_results = {}

        # Submit extraction tasks to executor
        futures = {}

        for extractor_name in extractors:
            # Skip if extractor not available
            if (
                extractor_name not in self.extractors
                or self.extractors[extractor_name] is None
            ):
                logger.warning(f"Extractor {extractor_name} not available")
                continue

            # Get extractor
            extractor = self.extractors[extractor_name]

            # Submit extraction task
            logger.info(f"Submitting extraction task for {extractor_name}")

            # Handle different parameter naming conventions
            if extractor_name in [
                "rule_based",
                "ner",
                "enhanced_rule_based",
                "optimized_ner",
            ]:
                # These extractors expect min_confidence and max_results
                future = self.executor.submit(
                    extractor.extract_techniques,
                    text=text,
                    min_confidence=threshold,
                    max_results=top_k,
                    job_id=job_id,  # Pass job_id for metrics recording
                )
            else:
                # Other extractors expect threshold and top_k
                future = self.executor.submit(
                    extractor.extract_techniques,
                    text=text,
                    threshold=threshold,
                    top_k=top_k,
                    job_id=job_id,  # Pass job_id for metrics recording
                )

            futures[extractor_name] = future

        # Collect results
        for extractor_name, future in futures.items():
            try:
                results = future.result()
                extractor_results[extractor_name] = results
            except Exception as e:
                logger.error(f"Error extracting techniques with {extractor_name}: {e}")
                extractor_results[extractor_name] = []

        # Use ensemble if requested
        if use_ensemble and extractor_results:
            # Use advanced ensemble
            techniques = self.ensemble.ensemble_extractors(
                text=text,
                extractor_results=extractor_results,
                threshold=threshold,
                max_results=top_k,
                job_id=job_id,  # Pass job_id for metrics recording
            )
        else:
            # Combine results without ensemble
            all_results = []
            for results in extractor_results.values():
                all_results.extend(results)

            # Remove duplicates, sort by confidence, and limit results
            seen_techniques = set()
            techniques = []

            for result in sorted(
                all_results, key=lambda x: x.get("confidence", 0), reverse=True
            ):
                tech_id = result.get("technique_id")
                if tech_id and tech_id not in seen_techniques:
                    seen_techniques.add(tech_id)
                    techniques.append(result)

                    if len(techniques) >= top_k:
                        break

        return {"techniques": techniques}

    def _add_context(self, response: Dict, techniques: List[Dict]) -> None:
        """
        Add contextual information to techniques

        Args:
            response: Response dictionary to update
            techniques: List of technique dictionaries
        """
        # Mark as including context
        response["context_included"] = True

        # Add context to each technique
        for technique in techniques:
            tech_id = technique.get("technique_id")
            if not tech_id:
                continue

            # Create context object
            context = {}

            # Add technique details
            if tech_id in self.techniques_data:
                tech_data = self.techniques_data[tech_id]

                # Add tactics
                context["tactics"] = tech_data.get("tactics", [])

                # Add platforms if available
                if "platforms" in tech_data:
                    context["platforms"] = tech_data["platforms"]

                # Add data sources if available
                if "data_sources" in tech_data:
                    context["data_sources"] = tech_data["data_sources"]

            # Add mitigations from Neo4j if available
            if self.use_neo4j and self.neo4j:
                try:
                    # Query for mitigations
                    mitigations_query = """
                    MATCH (t:AttackTechnique {technique_id: $technique_id})<-[:MITIGATES]-(m:AttackMitigation)
                    RETURN m.mitigation_id as id, m.name as name
                    """

                    mitigations_result = self.neo4j.run_query(
                        mitigations_query, {"technique_id": tech_id}
                    )

                    if mitigations_result:
                        context["mitigations"] = [dict(m) for m in mitigations_result]
                except Exception as e:
                    logger.error(f"Error getting mitigations: {e}")

            # Add similar techniques from Neo4j if available
            if self.use_neo4j and self.neo4j:
                try:
                    # Query for similar techniques
                    similar_query = """
                    MATCH (t:AttackTechnique {technique_id: $technique_id})-[r:RELATED_TO|SIMILAR_TO]-(related:AttackTechnique)
                    RETURN related.technique_id as technique_id, related.name as name, type(r) as relationship_type
                    LIMIT 5
                    """

                    similar_result = self.neo4j.run_query(
                        similar_query, {"technique_id": tech_id}
                    )

                    if similar_result:
                        context["similar_techniques"] = [
                            dict(t) for t in similar_result
                        ]
                except Exception as e:
                    logger.error(f"Error getting similar techniques: {e}")

            # Add context to technique
            technique["context"] = context

    def _add_relationships(self, response: Dict, techniques: List[Dict]) -> None:
        """
        Add relationship information to techniques

        Args:
            response: Response dictionary to update
            techniques: List of technique dictionaries
        """
        # Mark as including relationships
        response["relationships_included"] = True

        # Skip if Neo4j not available
        if not self.use_neo4j or not self.neo4j:
            return

        # Add relationships to each technique
        for technique in techniques:
            tech_id = technique.get("technique_id")
            if not tech_id:
                continue

            try:
                # Query for technique relationships
                query = """
                MATCH (t:AttackTechnique {technique_id: $technique_id})-[r]-(related)
                WHERE (related:AttackTechnique OR related:CAPEC OR related:CVE OR related:AttackSoftware)
                RETURN type(r) as relationship_type,
                       labels(related)[0] as related_type,
                       CASE 
                         WHEN 'AttackTechnique' IN labels(related) THEN related.technique_id
                         WHEN 'CAPEC' IN labels(related) THEN related.capec_id
                         WHEN 'CVE' IN labels(related) THEN related.cve_id
                         WHEN 'AttackSoftware' IN labels(related) THEN related.software_id
                         ELSE ''
                       END as related_id,
                       CASE 
                         WHEN related.name IS NOT NULL THEN related.name
                         ELSE ''
                       END as related_name
                LIMIT 15
                """

                results = self.neo4j.run_query(query, {"technique_id": tech_id})

                if results:
                    technique["relationships"] = [dict(r) for r in results]
            except Exception as e:
                logger.error(f"Error getting relationships: {e}")

    def _generate_navigator_layer(self, techniques: List[Dict]) -> Dict:
        """
        Generate MITRE Navigator layer from techniques

        Args:
            techniques: List of technique dictionaries

        Returns:
            Navigator layer dictionary
        """
        from datetime import datetime

        # Process techniques
        layer_techniques = []

        for technique in techniques:
            tech_id = technique.get("technique_id")
            confidence = technique.get("confidence", 0.5)
            method = technique.get("method", "unknown")

            # Skip invalid entries
            if not tech_id:
                continue

            # Generate color based on confidence
            color = self._get_color_from_score(confidence)

            # Create technique entry
            tech_entry = {
                "techniqueID": tech_id,
                "score": confidence,
                "color": color,
                "comment": f"Extracted using {method} method",
                "enabled": True,
                "metadata": [
                    {"name": "method", "value": method},
                    {"name": "confidence", "value": str(round(confidence * 100)) + "%"},
                ],
            }

            # Add component scores if available
            if "component_scores" in technique:
                for extractor, score in technique["component_scores"].items():
                    tech_entry["metadata"].append(
                        {
                            "name": f"{extractor}_score",
                            "value": str(round(score * 100)) + "%",
                        }
                    )

            # Add matched keywords if available
            if "matched_keywords" in technique and technique["matched_keywords"]:
                keywords = technique["matched_keywords"]
                tech_entry["metadata"].append(
                    {
                        "name": "matched_keywords",
                        "value": ", ".join(keywords[:5])
                        + ("..." if len(keywords) > 5 else ""),
                    }
                )

            # Add CVE if available
            if "cve_id" in technique:
                tech_entry["metadata"].append(
                    {"name": "cve", "value": technique["cve_id"]}
                )

            layer_techniques.append(tech_entry)

        # Create layer
        layer = {
            "name": "Extraction Results",
            "versions": {
                "attack": "13",
                "navigator": "4.8.0",
                "layer": "4.4",
            },
            "domain": "enterprise-attack",
            "description": f"Layer generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "techniques": layer_techniques,
            "gradient": {
                "colors": ["#ffffdd", "#ff6666"],
                "minValue": 0,
                "maxValue": 1,
            },
            "legendItems": [
                {"label": "Low Confidence", "color": "#ffffdd"},
                {"label": "Medium Confidence", "color": "#ffb366"},
                {"label": "High Confidence", "color": "#ff6666"},
            ],
            "metadata": [
                {"name": "generated_by", "value": "Enhanced ATT&CK Extractor"},
                {"name": "generated_at", "value": datetime.now().isoformat()},
            ],
        }

        return layer

    def _get_color_from_score(self, score: float) -> str:
        """
        Get color for MITRE Navigator based on confidence score

        Args:
            score: Confidence score

        Returns:
            Color string in hex format
        """
        if score >= 0.8:
            return "#ff6666"  # Red - high confidence
        elif score >= 0.5:
            return "#ffb366"  # Orange - medium confidence
        elif score >= 0.3:
            return "#ffff99"  # Yellow - low confidence
        else:
            return "#ffffdd"  # Light yellow - very low confidence

    def extract_techniques_batch(
        self,
        texts: List[str],
        extractors: List[str] = None,
        threshold: float = 0.2,
        top_k: int = 10,
        use_ensemble: bool = True,
        include_context: bool = False,
        include_relationships: bool = False,
        return_navigator_layer: bool = False,
        user_id: str = None,
        tier: str = "basic",
        request_id: str = None,
        batch_size: int = 5,
    ) -> List[Dict]:
        """
        Extract ATT&CK techniques from multiple texts in batch

        Args:
            texts: List of input texts
            extractors: List of extractors to use
            threshold: Minimum confidence threshold
            top_k: Maximum number of results
            use_ensemble: Whether to use ensemble method
            include_context: Whether to include contextual information
            include_relationships: Whether to include technique relationships
            return_navigator_layer: Whether to return MITRE Navigator layer
            user_id: User identifier for caching and rate limiting
            tier: User tier for rate limiting
            request_id: Request identifier for tracking
            batch_size: Size of batches for processing

        Returns:
            List of dictionaries with extraction results
        """
        # Check if texts are valid
        if not texts:
            return []

        # Process all texts with multi-threading
        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
            )

            # Submit batch for processing
            futures = []

            for j, text in enumerate(batch):
                batch_request_id = f"{request_id}_{i+j}" if request_id else None

                future = self.executor.submit(
                    self.extract_techniques,
                    text=text,
                    extractors=extractors,
                    threshold=threshold,
                    top_k=top_k,
                    use_ensemble=use_ensemble,
                    include_context=include_context,
                    include_relationships=include_relationships,
                    return_navigator_layer=return_navigator_layer,
                    user_id=user_id,
                    tier=tier,
                    request_id=batch_request_id,
                )

                futures.append(future)

            # Collect results as they complete
            for future in futures:
                results.append(future.result())

        return results

    def process_feedback(self, feedback_data: Dict) -> bool:
        """
        Process feedback for learning and improvement

        Args:
            feedback_data: Dictionary with feedback data

        Returns:
            Whether feedback processing was successful
        """
        try:
            # Update ensemble with feedback
            self.ensemble.update_from_feedback(feedback_data)

            # Invalidate cache for related entries if needed
            if self.use_caching and self.cache:
                # Get analysis text if available
                analysis_id = feedback_data.get("analysis_id")
                if analysis_id:
                    # This would require database access to get the original text
                    # For now, we'll just invalidate cache entries with matching technique IDs
                    technique_id = feedback_data.get("technique_id")
                    if technique_id:
                        # This is a rough invalidation - in a real system, we'd be more precise
                        self.cache.invalidate(pattern=f"*_{technique_id}*")

            return True
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return False

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            Dictionary of cache statistics
        """
        if self.use_caching and self.cache:
            return self.cache.get_stats()
        else:
            return {"cache_enabled": False}

    def cleanup_cache(self) -> int:
        """
        Clean up expired cache entries

        Returns:
            Number of entries removed
        """
        if self.use_caching and self.cache:
            return self.cache.cleanup()
        else:
            return 0

    def unload_extractors(self) -> None:
        """Unload extractors to free memory"""
        for extractor_name, extractor in self.extractors.items():
            if extractor and hasattr(extractor, "unload_model"):
                try:
                    extractor.unload_model()
                    logger.info(f"Unloaded {extractor_name} extractor")
                except Exception as e:
                    logger.error(f"Error unloading {extractor_name} extractor: {e}")
