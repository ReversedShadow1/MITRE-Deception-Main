"""
Optimized Neural Extractor for ATT&CK Techniques
----------------------------------------------
Implements efficient neural extraction with model quantization, batching,
and memory-efficient processing for transformer models.
"""

import logging
import os
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from src.database.postgresql import get_db

logger = logging.getLogger("OptimizedNeuralExtractor")


class ModelCache:
    """
    Shared model cache for efficient model reuse across requests
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = ModelCache()
        return cls._instance

    def __init__(self):
        """Initialize model cache"""
        self.models = {}
        self.tokenizers = {}
        self.last_used = {}
        self.max_cache_size = 3  # Maximum number of models to keep in cache

    def get_model(self, model_id: str) -> Tuple[Any, Any]:
        """
        Get model and tokenizer from cache

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (model, tokenizer)
        """
        if model_id in self.models and model_id in self.tokenizers:
            # Update last used timestamp
            self.last_used[model_id] = time.time()
            return self.models[model_id], self.tokenizers[model_id]
        return None, None

    def add_model(self, model_id: str, model: Any, tokenizer: Any) -> None:
        """
        Add model to cache

        Args:
            model_id: Model identifier
            model: Model object
            tokenizer: Tokenizer object
        """
        # Check if cache is full
        if len(self.models) >= self.max_cache_size:
            # Remove least recently used model
            lru_model = min(self.last_used.items(), key=lambda x: x[1])[0]
            self.remove_model(lru_model)

        # Add model to cache
        self.models[model_id] = model
        self.tokenizers[model_id] = tokenizer
        self.last_used[model_id] = time.time()

    def remove_model(self, model_id: str) -> None:
        """
        Remove model from cache

        Args:
            model_id: Model identifier
        """
        if model_id in self.models:
            # Remove from cache
            del self.models[model_id]
            del self.tokenizers[model_id]
            del self.last_used[model_id]

            # Force garbage collection
            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class OptimizedNeuralExtractor:
    """
    Base class for optimized neural extractors with shared functionality
    """

    def __init__(
        self,
        techniques_data: Dict,
        technique_keywords: Dict,
        model_name: str,
        model_type: str,
        cache_dir: str,
        use_gpu: bool = True,
        use_quantization: bool = True,
        batch_size: int = 16,
        use_model_cache: bool = True,
        neo4j_connector=None,
    ):
        """
        Initialize optimized neural extractor

        Args:
            techniques_data: Dictionary of technique data
            technique_keywords: Dictionary of technique keywords
            model_name: Name or path of the model
            model_type: Type of model ('ner' or 'embedding')
            cache_dir: Directory for model caching
            use_gpu: Whether to use GPU acceleration
            use_quantization: Whether to use model quantization
            batch_size: Size of batches for processing
            use_model_cache: Whether to use shared model cache
            neo4j_connector: Optional Neo4j connector
        """
        self.techniques_data = techniques_data
        self.technique_keywords = technique_keywords
        self.model_name = model_name
        self.model_type = model_type
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.use_quantization = use_quantization
        self.batch_size = batch_size
        self.use_model_cache = use_model_cache
        self.neo4j_connector = neo4j_connector

        # Model components
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if self.use_gpu else "cpu"

        # Status flags
        self.is_loaded = False

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Log initialization
        logger.info(f"Initialized optimized neural extractor with model {model_name}")
        logger.info(f"Using device: {self.device}")

        # Set model cache ID
        self.model_cache_id = f"{model_name}_{model_type}"

        if self.use_quantization:
            self.model_cache_id += "_quantized"

    def load_model(self) -> bool:
        """
        Load the model with optimizations

        Returns:
            Whether loading was successful
        """
        if self.is_loaded:
            return True

        # Check if model is in cache
        if self.use_model_cache:
            cache = ModelCache.get_instance()
            model, tokenizer = cache.get_model(self.model_cache_id)

            if model is not None and tokenizer is not None:
                self.model = model
                self.tokenizer = tokenizer
                self.is_loaded = True
                logger.info(f"Loaded model {self.model_name} from cache")
                return True

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir, use_fast=True
            )

            # Load model with appropriate class based on type
            if self.model_type == "ner":
                model_class = AutoModelForTokenClassification
            else:  # embedding
                model_class = AutoModel

            # Load the model
            load_options = {"cache_dir": self.cache_dir}

            # Add quantization options if enabled
            if self.use_quantization:
                load_options["torch_dtype"] = torch.float16
                load_options["low_cpu_mem_usage"] = True

            self.model = model_class.from_pretrained(self.model_name, **load_options)

            # Move to device
            self.model = self.model.to(self.device)

            # Apply further quantization if needed (int8 quantization)
            if self.use_quantization and self.device == "cuda":
                try:
                    # Only import if needed
                    import torch.quantization

                    # Set to evaluation mode
                    self.model.eval()

                    # Quantize model
                    self.model = torch.quantization.quantize_dynamic(
                        self.model,
                        {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d},
                        dtype=torch.qint8,
                    )

                    logger.info(f"Applied int8 quantization to model {self.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to apply int8 quantization: {e}")

            # Add to cache if enabled
            if self.use_model_cache:
                cache = ModelCache.get_instance()
                cache.add_model(self.model_cache_id, self.model, self.tokenizer)

            self.is_loaded = True
            logger.info(f"Successfully loaded model {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            return False

    def unload_model(self) -> None:
        """Unload the model to free memory"""
        if not self.is_loaded:
            return

        # If using model cache, just mark as unloaded but keep in cache
        if self.use_model_cache:
            self.is_loaded = False
            return

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

            self.is_loaded = False
            logger.info(f"Unloaded model {self.model_name}")

        except Exception as e:
            logger.error(f"Error unloading model {self.model_name}: {e}")

    def _segment_text(self, text: str, max_length: int = 512) -> List[str]:
        """
        Segment text into chunks for efficient processing

        Args:
            text: Input text
            max_length: Maximum segment length in tokens

        Returns:
            List of text segments
        """
        # Simple segmentation for now - split by newlines and then by size
        segments = []
        paragraphs = text.split("\n")

        current_segment = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph would exceed max length
            # This is an approximation as we don't tokenize yet
            if (
                len(current_segment) + len(paragraph) > max_length * 4
            ):  # Rough char to token ratio
                if current_segment:
                    segments.append(current_segment)

                # Check if paragraph itself is too long
                if len(paragraph) > max_length * 4:
                    # Split paragraph into sentences
                    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                    current_segment = ""

                    for sentence in sentences:
                        if len(current_segment) + len(sentence) > max_length * 4:
                            if current_segment:
                                segments.append(current_segment)
                            current_segment = sentence
                        else:
                            if current_segment:
                                current_segment += " " + sentence
                            else:
                                current_segment = sentence
                else:
                    current_segment = paragraph
            else:
                if current_segment:
                    current_segment += " " + paragraph
                else:
                    current_segment = paragraph

        # Add the last segment if there is one
        if current_segment:
            segments.append(current_segment)

        return segments


class OptimizedNERExtractor(OptimizedNeuralExtractor):
    """
    Optimized NER extractor using transformer models with quantization and batching
    """

    def __init__(
        self,
        techniques_data: Dict,
        technique_keywords: Dict,
        model_name: str = "CyberPeace-Institute/SecureBERT-NER",
        cache_dir: str = "models/ner_optimized",
        use_gpu: bool = True,
        use_quantization: bool = True,
        batch_size: int = 16,
        use_model_cache: bool = True,
        confidence_calibration: bool = True,
        neo4j_connector=None,
    ):
        """
        Initialize optimized NER extractor

        Args:
            techniques_data: Dictionary of technique data
            technique_keywords: Dictionary of technique keywords
            model_name: Name or path of the NER model
            cache_dir: Directory for model caching
            use_gpu: Whether to use GPU acceleration
            use_quantization: Whether to use model quantization
            batch_size: Size of batches for processing
            use_model_cache: Whether to use shared model cache
            confidence_calibration: Whether to calibrate confidence scores
            neo4j_connector: Optional Neo4j connector
        """
        super().__init__(
            techniques_data=techniques_data,
            technique_keywords=technique_keywords,
            model_name=model_name,
            model_type="ner",
            cache_dir=cache_dir,
            use_gpu=use_gpu,
            use_quantization=use_quantization,
            batch_size=batch_size,
            use_model_cache=use_model_cache,
            neo4j_connector=neo4j_connector,
        )

        self.confidence_calibration = confidence_calibration
        self.ner_pipeline = None

        # Define entity type to technique mapping for better accuracy
        self.entity_type_technique_affinity = {
            "ATTACK-PATTERN": ["T1190", "T1133", "T1566", "T1566.001", "T1078"],
            "MALWARE": ["T1059", "T1204", "T1027", "T1027.002", "T1055"],
            "TOOL": ["T1059.001", "T1059.003", "T1059.005", "T1059.006", "T1021.006"],
            "VULNERABILITY": ["T1190", "T1068", "T1211", "T1212", "T1203"],
            "FILE": ["T1005", "T1074", "T1083", "T1564", "T1560"],
            "SYSTEM": ["T1082", "T1018", "T1016", "T1049", "T1033"],
            "NETWORK": ["T1046", "T1049", "T1071", "T1095", "T1571"],
            "USER-ACCOUNT": ["T1078", "T1136", "T1087", "T1098", "T1003.001"],
            "PROTOCOL": ["T1071.001", "T1071.002", "T1071.003", "T1071.004", "T1095"],
            "CREDENTIAL": ["T1003", "T1110", "T1555", "T1552", "T1556"],
        }

    def _setup_ner_pipeline(self) -> bool:
        """
        Set up the NER pipeline

        Returns:
            Whether setup was successful
        """
        if not self.is_loaded:
            if not self.load_model():
                return False

        try:
            # Create optimized NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.use_gpu else -1,
                aggregation_strategy="simple",
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set up NER pipeline: {e}")
            return False

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text with optimized processing

        Args:
            text: Input text

        Returns:
            Dictionary mapping entity types to lists of entities
        """
        if not self.is_loaded:
            if not self.load_model():
                logger.error("NER model not loaded")
                return {}

        if self.ner_pipeline is None:
            if not self._setup_ner_pipeline():
                logger.error("Failed to set up NER pipeline")
                return {}

        try:
            # Segment text for efficient processing
            segments = self._segment_text(text, max_length=512)

            all_entities = []

            # Process segments in batches
            for i in range(0, len(segments), self.batch_size):
                batch = segments[i : i + self.batch_size]

                # Process batch with NER pipeline
                batch_entities = self.ner_pipeline(batch)

                # Flatten if batch size is 1
                if len(batch) == 1:
                    batch_entities = [batch_entities]

                # Add to all entities
                all_entities.extend(batch_entities)

            # Group entities by type
            grouped_entities = {}

            for segment_entities in all_entities:
                for entity in segment_entities:
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
            logger.error(f"Error extracting entities: {e}")
            return {}

    '''def extract_techniques(
        self, 
        text: str, 
        min_confidence: float = 0.1, 
        max_results: int = 10
    ) -> List[Dict]:
        """
        Extract techniques using optimized NER
        
        Args:
            text: Input text
            min_confidence: Minimum confidence threshold
            max_results: Maximum number of results
            
        Returns:
            List of technique matches with confidence scores
        """
        # Extract entities
        entities = self.extract_entities(text)
        
        if not entities:
            logger.info("No entities found in text")
            return []
        
        # Get all entities as a flat list
        all_entities = []
        entity_types = {}  # Track entity type for each entity
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                all_entities.append(entity)
                entity_types[entity] = entity_type
        
        logger.info(f"Found {len(all_entities)} entities across {len(entities)} entity types")
        
        # Match entities to techniques
        technique_matches = {}
        
        for entity in all_entities:
            entity_lower = entity.lower()
            entity_type = entity_types.get(entity, "UNKNOWN")
            
            # Check for direct matches in technique keywords
            for tech_id, keywords in self.technique_keywords.items():
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    
                    # Check if entity contains keyword or vice versa
                    if keyword_lower in entity_lower or entity_lower in keyword_lower:
                        if tech_id not in technique_matches:
                            technique_matches[tech_id] = {
                                "count": 0, 
                                "entities": [],
                                "entity_types": set()
                            }
                        
                        technique_matches[tech_id]["count"] += 1
                        
                        if entity not in technique_matches[tech_id]["entities"]:
                            technique_matches[tech_id]["entities"].append(entity)
                            technique_matches[tech_id]["entity_types"].add(entity_type)
        
        # Convert matches to results
        results = []
        
        for tech_id, match_data in technique_matches.items():
            # Calculate confidence based on match count and unique entities
            match_count = match_data["count"]
            unique_entities = len(match_data["entities"])
            entity_types = match_data["entity_types"]
            
            # Base confidence formula: base + boost for multiple entities
            base_confidence = min(0.4 + (unique_entities * 0.05), 0.75)
            
            # Apply entity type affinity boost
            affinity_boost = 1.0
            
            for entity_type in entity_types:
                if entity_type in self.entity_type_technique_affinity:
                    affinity_techniques = self.entity_type_technique_affinity[entity_type]
                    
                    # If technique has affinity with this entity type, boost confidence
                    if tech_id in affinity_techniques:
                        affinity_boost = 1.2
                        break
                    # If parent technique has affinity, smaller boost
                    elif any(tech_id.startswith(t.split('.')[0]) for t in affinity_techniques if '.' in t):
                        affinity_boost = 1.1
                        break
            
            # Apply calibration if enabled
            if self.confidence_calibration:
                # Adjust confidence based on empirical calibration
                # This helps prevent overconfidence in the NER results
                calibrated_confidence = 0.3 + (base_confidence * 0.5)
            else:
                calibrated_confidence = base_confidence
            
            # Final confidence with boosts
            final_confidence = min(calibrated_confidence * affinity_boost, 0.9)
            
            result = {
                "technique_id": tech_id,
                "confidence": final_confidence,
                "match_count": match_count,
                "matched_entities": match_data["entities"],
                "entity_types": list(entity_types),
                "method": "optimized_ner",
            }
            
            # Add technique name if available
            if tech_id in self.techniques_data:
                result["name"] = self.techniques_data[tech_id].get("name", "")
            
            results.append(result)
        
        # Filter by confidence threshold
        results = [r for r in results if r["confidence"] >= min_confidence]
        
        # Sort by confidence and limit results
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:max_results]'''

    # Add this to OptimizedNERExtractor in src/api/v2/routes/Optimized_Neural_Extractor.py

    def extract_techniques(
        self,
        text: str,
        min_confidence: float = 0.1,
        max_results: int = 10,
        job_id: str = None,
    ) -> List[Dict]:
        """
        Extract techniques using optimized NER with metrics recording

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
                "batch_size": self.batch_size,
                "use_quantization": self.use_quantization,
                "confidence_calibration": self.confidence_calibration,
            }
            extractor_id = metrics_recorder.record_extractor_result(
                extractor_name="optimized_ner",
                raw_input=text,
                raw_output={},  # Will update this later
                execution_time_ms=0,  # Will update this later
                parameters=parameters,
            )

            # Record model details
            metrics_recorder.record_model_metrics(
                extractor_id=extractor_id,
                model_name=self.model_name,
                device_used=self.device,
                batch_size=self.batch_size,
                quantization_used=self.use_quantization,
            )

            # Record text segment
            metrics_recorder.record_text_segment(text=text, index=0)

        # Load model if needed
        load_start = time.time()
        if not self.is_loaded:
            self.load_model()

            # Record model loading performance
            if metrics_recorder and extractor_id:
                load_time_ms = int((time.time() - load_start) * 1000)
                metrics_recorder.record_performance_benchmark(
                    extractor_name="optimized_ner",
                    operation_type="model_loading",
                    execution_time_ms=load_time_ms,
                )

        # Extract entities
        entity_start = time.time()
        entities = self.extract_entities(text)
        entity_time = int((time.time() - entity_start) * 1000)

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

            total_entity_count = sum(
                len(entity_list) for entity_type, entity_list in entities.items()
            )

            # Record NER details
            metrics_recorder.record_ner_details(
                extractor_id=extractor_id,
                entity_count=total_entity_count,
                entity_types=entity_type_counts,
                model_name=self.model_name,
                aggregation_strategy="simple",
                tokenizer_max_length=512,
            )

            # Record entity extraction performance
            metrics_recorder.record_performance_benchmark(
                extractor_name="optimized_ner",
                operation_type="entity_extraction",
                execution_time_ms=entity_time,
                input_size=len(text),
            )

            # Record each entity found
            for entity_type, entity_list in entities.items():
                for i, entity_text in enumerate(entity_list):
                    # Find position in text
                    pos = text.lower().find(entity_text.lower())

                    entity_data = {
                        "text": entity_text,
                        "type": entity_type,
                        "start_offset": pos if pos >= 0 else None,
                        "end_offset": pos + len(entity_text) if pos >= 0 else None,
                        "confidence": None,  # NER pipeline doesn't always provide per-entity confidence
                    }

                    metrics_recorder.record_entities(
                        extractor_id=extractor_id, entities=[entity_data]
                    )

        # Get all entities as a flat list
        all_entities = []
        entity_types = {}  # Track entity type for each entity

        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                all_entities.append(entity)
                entity_types[entity] = entity_type

        logger.info(
            f"Found {len(all_entities)} entities across {len(entities)} entity types"
        )

        # Match entities to techniques
        match_start = time.time()
        technique_matches = {}

        for entity in all_entities:
            entity_lower = entity.lower()
            entity_type = entity_types.get(entity, "UNKNOWN")

            # Check for direct matches in technique keywords
            for tech_id, keywords in self.technique_keywords.items():
                for keyword in keywords:
                    keyword_lower = keyword.lower()

                    # Check if entity contains keyword or vice versa
                    if keyword_lower in entity_lower or entity_lower in keyword_lower:
                        if tech_id not in technique_matches:
                            technique_matches[tech_id] = {
                                "count": 0,
                                "entities": [],
                                "entity_types": set(),
                            }

                        technique_matches[tech_id]["count"] += 1

                        if entity not in technique_matches[tech_id]["entities"]:
                            technique_matches[tech_id]["entities"].append(entity)
                            technique_matches[tech_id]["entity_types"].add(entity_type)

        match_time = int((time.time() - match_start) * 1000)

        # Record technique matching performance
        if metrics_recorder and extractor_id:
            metrics_recorder.record_performance_benchmark(
                extractor_name="optimized_ner",
                operation_type="technique_matching",
                execution_time_ms=match_time,
                input_size=len(all_entities),
            )

        # Convert matches to results
        results = []

        for tech_id, match_data in technique_matches.items():
            # Calculate confidence based on match count and unique entities
            match_count = match_data["count"]
            unique_entities = len(match_data["entities"])
            entity_types = match_data["entity_types"]

            # Base confidence formula: base + boost for multiple entities
            base_confidence = min(0.4 + (unique_entities * 0.05), 0.75)

            # Apply entity type affinity boost
            affinity_boost = 1.0

            for entity_type in entity_types:
                if entity_type in self.entity_type_technique_affinity:
                    affinity_techniques = self.entity_type_technique_affinity[
                        entity_type
                    ]

                    # If technique has affinity with this entity type, boost confidence
                    if tech_id in affinity_techniques:
                        affinity_boost = 1.2
                        break
                    # If parent technique has affinity, smaller boost
                    elif any(
                        tech_id.startswith(t.split(".")[0])
                        for t in affinity_techniques
                        if "." in t
                    ):
                        affinity_boost = 1.1
                        break

            # Apply calibration if enabled
            if self.confidence_calibration:
                # Adjust confidence based on empirical calibration
                # This helps prevent overconfidence in the NER results
                calibrated_confidence = 0.3 + (base_confidence * 0.5)
            else:
                calibrated_confidence = base_confidence

            # Final confidence with boosts
            final_confidence = min(calibrated_confidence * affinity_boost, 0.9)

            result = {
                "technique_id": tech_id,
                "confidence": final_confidence,
                "match_count": match_count,
                "matched_entities": match_data["entities"],
                "entity_types": list(entity_types),
                "method": "optimized_ner",
            }

            # Add technique name if available
            if tech_id in self.techniques_data:
                result["name"] = self.techniques_data[tech_id].get("name", "")

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


class OptimizedEmbeddingExtractor(OptimizedNeuralExtractor):
    """
    Optimized embedding extractor using transformer models with
    quantization, batching, and efficient similarity search
    """

    def __init__(
        self,
        techniques_data: Dict,
        technique_keywords: Dict,
        model_name: str = "BAAI/bge-large-en-v1.5",
        cache_dir: str = "models/embedding_optimized",
        embeddings_dir: str = "models/embeddings_optimized",
        use_gpu: bool = True,
        use_quantization: bool = True,
        batch_size: int = 16,
        use_model_cache: bool = True,
        cache_embeddings: bool = True,
        use_approximate_search: bool = True,
        neo4j_connector=None,
    ):
        """
        Initialize optimized embedding extractor

        Args:
            techniques_data: Dictionary of technique data
            technique_keywords: Dictionary of technique keywords
            model_name: Name or path of the embedding model
            cache_dir: Directory for model caching
            embeddings_dir: Directory for storing embeddings
            use_gpu: Whether to use GPU acceleration
            use_quantization: Whether to use model quantization
            batch_size: Size of batches for processing
            use_model_cache: Whether to use shared model cache
            cache_embeddings: Whether to cache technique embeddings
            use_approximate_search: Whether to use approximate similarity search
            neo4j_connector: Optional Neo4j connector
        """
        super().__init__(
            techniques_data=techniques_data,
            technique_keywords=technique_keywords,
            model_name=model_name,
            model_type="embedding",
            cache_dir=cache_dir,
            use_gpu=use_gpu,
            use_quantization=use_quantization,
            batch_size=batch_size,
            use_model_cache=use_model_cache,
            neo4j_connector=neo4j_connector,
        )

        self.embeddings_dir = embeddings_dir
        self.cache_embeddings = cache_embeddings
        self.use_approximate_search = use_approximate_search

        # Embedding cache
        self.embeddings_cache_path = os.path.join(
            embeddings_dir, "technique_embeddings.npy"
        )
        self.embedding_ids_path = os.path.join(
            embeddings_dir, "technique_embedding_ids.json"
        )
        self.technique_embeddings = None
        self.embedding_tech_ids = []

        # Create embeddings directory
        os.makedirs(embeddings_dir, exist_ok=True)

        # Approximate search index
        self.search_index = None

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

            with open(self.embedding_ids_path, "r") as f:
                import json

                self.embedding_tech_ids = json.load(f)

            logger.info(
                f"Loaded {len(self.embedding_tech_ids)} cached technique embeddings"
            )

            # Build search index if using approximate search
            if self.use_approximate_search:
                self._build_search_index()

            return True

        except Exception as e:
            logger.error(f"Failed to load cached embeddings: {e}")
            return False

    def _build_search_index(self) -> None:
        """Build approximate search index for efficient similarity search"""
        if self.technique_embeddings is None or len(self.technique_embeddings) == 0:
            logger.warning("No embeddings available to build search index")
            return

        try:
            # Try to import faiss for approximate search
            import faiss

            # Get embedding dimension
            dim = self.technique_embeddings.shape[1]

            # Create appropriate index based on size
            if len(self.technique_embeddings) < 10000:
                # For smaller datasets, use exact search with L2 distance
                self.search_index = faiss.IndexFlatL2(dim)
            else:
                # For larger datasets, use approximate search with IVF
                nlist = min(int(np.sqrt(len(self.technique_embeddings))), 100)
                quantizer = faiss.IndexFlatL2(dim)
                self.search_index = faiss.IndexIVFFlat(
                    quantizer, dim, nlist, faiss.METRIC_L2
                )
                self.search_index.train(self.technique_embeddings)

            # Add embeddings to index
            self.search_index.add(self.technique_embeddings)
            logger.info(
                f"Built search index for {len(self.technique_embeddings)} embeddings"
            )

        except ImportError:
            logger.warning("Faiss not available, falling back to exact search")
            self.use_approximate_search = False
        except Exception as e:
            logger.error(f"Failed to build search index: {e}")
            self.use_approximate_search = False

    def _generate_technique_embeddings(self) -> None:
        """Generate embeddings for all techniques with efficient batch processing"""
        if not self.is_loaded:
            if not self.load_model():
                logger.error("Model not loaded, cannot generate embeddings")
                return

        try:
            tech_ids = []
            embeddings = []

            # Create rich text representations for all techniques
            technique_texts = []

            for tech_id, tech_data in self.techniques.items():
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

                technique_texts.append((tech_id, text))

            # Process techniques in batches
            for i in range(0, len(technique_texts), self.batch_size):
                batch = technique_texts[i : i + self.batch_size]
                batch_ids = [item[0] for item in batch]
                batch_texts = [item[1] for item in batch]

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
                    import json

                    json.dump(self.embedding_tech_ids, f)

                logger.info(f"Cached embeddings for {len(tech_ids)} techniques")

            # Build search index if using approximate search
            if self.use_approximate_search:
                self._build_search_index()

        except Exception as e:
            logger.error(f"Error generating technique embeddings: {e}")

    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a list of texts with efficient batch processing

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

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                sub_batch = texts[i : i + self.batch_size]

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
            logger.error(f"Error getting embeddings: {e}")
            return [np.zeros(768) for _ in texts]  # Return dummy embeddings

    @lru_cache(maxsize=128)
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors with caching

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        # Convert to numpy arrays if they're not already
        if isinstance(a, list):
            a = np.array(a)
        if isinstance(b, list):
            b = np.array(b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0

        return max(0, min(1, np.dot(a, b) / (norm_a * norm_b)))

    '''def extract_techniques(
        self, 
        text: str, 
        threshold: float = 0.5, 
        top_k: int = 10
    ) -> List[Dict]:
        """
        Extract techniques using semantic search with optimizations
        
        Args:
            text: Input text
            threshold: Minimum similarity threshold
            top_k: Maximum number of results
            
        Returns:
            List of technique matches with scores
        """
        # Load or generate technique embeddings if needed
        if self.technique_embeddings is None or len(self.embedding_tech_ids) == 0:
            if not self._load_technique_embeddings():
                logger.info("Generating technique embeddings...")
                self._generate_technique_embeddings()
        
        if self.technique_embeddings is None or len(self.embedding_tech_ids) == 0:
            logger.error("No technique embeddings available")
            return []
        
        try:
            # Segment text for better results with long inputs
            segments = self._segment_text(text, max_length=512)
            
            # Get embeddings for all segments
            segment_embeddings = self._get_embeddings(segments)
            
            # Combine segment results
            all_similarities = []
            
            # Find similar techniques for each segment
            for segment_embedding in segment_embeddings:
                # Use different search methods based on configuration
                if self.use_approximate_search and self.search_index is not None:
                    # Use approximate search with faiss
                    import faiss
                    
                    # Prepare query
                    query = np.array([segment_embedding], dtype=np.float32)
                    
                    # Search
                    D, I = self.search_index.search(query, min(top_k * 2, len(self.technique_embeddings)))
                    
                    # Convert to similarities
                    for i, idx in enumerate(I[0]):
                        if idx < len(self.embedding_tech_ids) and idx >= 0:
                            tech_id = self.embedding_tech_ids[idx]
                            # Convert L2 distance to similarity score
                            similarity = 1.0 / (1.0 + D[0][i])
                            all_similarities.append((tech_id, similarity))
                else:
                    # Use exact search with cosine similarity
                    for i, tech_id in enumerate(self.embedding_tech_ids):
                        if i >= len(self.technique_embeddings):
                            continue
                        
                        tech_embedding = self.technique_embeddings[i]
                        similarity = self._cosine_similarity(segment_embedding, tech_embedding)
                        
                        all_similarities.append((tech_id, similarity))
            
            # Aggregate similarities across segments
            technique_scores = {}
            
            for tech_id, similarity in all_similarities:
                if tech_id not in technique_scores or similarity > technique_scores[tech_id]:
                    technique_scores[tech_id] = similarity
            
            # Filter by threshold
            filtered_similarities = [
                (tech_id, sim) for tech_id, sim in technique_scores.items() if sim >= threshold
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
                    "method": "optimized_semantic",
                }
                
                # Add technique name if available
                if tech_id in self.techniques_data:
                    result["name"] = self.techniques_data[tech_id].get("name", "")
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting techniques with semantic search: {e}")
            return []'''

    # Add this to OptimizedEmbeddingExtractor in src/api/v2/routes/Optimized_Neural_Extractor.py

    def extract_techniques(
        self, text: str, threshold: float = 0.5, top_k: int = 10, job_id: str = None
    ) -> List[Dict]:
        """
        Extract techniques using optimized semantic search with metrics recording

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

        # Load or generate technique embeddings if needed
        if self.technique_embeddings is None or len(self.embedding_tech_ids) == 0:
            if not self._load_technique_embeddings():
                logger.info("Generating technique embeddings...")
                self._generate_technique_embeddings()

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
                "batch_size": self.batch_size,
                "use_quantization": self.use_quantization,
                "use_approximate_search": self.use_approximate_search,
            }
            extractor_id = metrics_recorder.record_extractor_result(
                extractor_name="optimized_semantic",
                raw_input=text,
                raw_output={},  # Will update this later
                execution_time_ms=0,  # Will update this later
                parameters=parameters,
            )

            # Record model details
            metrics_recorder.record_model_metrics(
                extractor_id=extractor_id,
                model_name=self.model_name,
                device_used=self.device,
                batch_size=self.batch_size,
                quantization_used=self.use_quantization,
            )

            # Record text segment
            metrics_recorder.record_text_segment(text=text, index=0)

        # Load model if needed
        load_start = time.time()
        if not self.is_loaded:
            self.load_model()

            # Record model loading performance
            if metrics_recorder and extractor_id:
                load_time_ms = int((time.time() - load_start) * 1000)
                metrics_recorder.record_performance_benchmark(
                    extractor_name="optimized_semantic",
                    operation_type="model_loading",
                    execution_time_ms=load_time_ms,
                )

        try:
            # Segment text for better results with long inputs
            segment_start = time.time()
            segments = self._segment_text(text, max_length=512)
            segment_time = int((time.time() - segment_start) * 1000)

            # Record text segmentation performance
            if metrics_recorder and extractor_id:
                metrics_recorder.record_performance_benchmark(
                    extractor_name="optimized_semantic",
                    operation_type="text_segmentation",
                    execution_time_ms=segment_time,
                    input_size=len(text),
                )

                # Record segments
                for i, segment in enumerate(segments):
                    metrics_recorder.record_text_segment(
                        text=segment,
                        index=i + 1,  # Start from 1 as the full text is index 0
                    )

            # Get embeddings for all segments
            embed_start = time.time()
            segment_embeddings = self._get_embeddings(segments)
            embed_time = int((time.time() - embed_start) * 1000)

            # Record embedding computation performance
            if metrics_recorder and extractor_id:
                metrics_recorder.record_performance_benchmark(
                    extractor_name="optimized_semantic",
                    operation_type="text_embedding",
                    execution_time_ms=embed_time,
                    input_size=sum(len(s) for s in segments),
                    throughput_tokens_per_second=sum(len(s) for s in segments)
                    / (embed_time / 1000)
                    if embed_time > 0
                    else None,
                )

                # Record embeddings
                for i, (segment, embedding) in enumerate(
                    zip(segments, segment_embeddings)
                ):
                    segment_id = metrics_recorder.record_text_segment(
                        text=segment,
                        index=i + 1,  # Start from 1 as the full text is index 0
                        embedding=embedding,
                    )

                    # Record embedding details
                    metrics_recorder.record_embedding_details(
                        extractor_id=extractor_id,
                        text_segment_id=segment_id,
                        embedding_type="query_segment",
                        embedding_model=self.model_name,
                        embedding_dimension=len(embedding)
                        if embedding is not None
                        else None,
                        normalization_applied=True,
                        cache_hit=False,
                        approximate_search_used=self.use_approximate_search,
                    )

            # Combine segment results
            all_similarities = []

            # Find similar techniques for each segment
            similar_start = time.time()
            for segment_embedding in segment_embeddings:
                # Use different search methods based on configuration
                if self.use_approximate_search and self.search_index is not None:
                    # Use approximate search with faiss

                    # Prepare query
                    query = np.array([segment_embedding], dtype=np.float32)

                    # Search
                    D, I = self.search_index.search(
                        query, min(top_k * 2, len(self.technique_embeddings))
                    )

                    # Convert to similarities
                    for i, idx in enumerate(I[0]):
                        if idx < len(self.embedding_tech_ids) and idx >= 0:
                            tech_id = self.embedding_tech_ids[idx]
                            # Convert L2 distance to similarity score
                            similarity = 1.0 / (1.0 + D[0][i])
                            all_similarities.append((tech_id, similarity))
                else:
                    # Use exact search with cosine similarity
                    for i, tech_id in enumerate(self.embedding_tech_ids):
                        if i >= len(self.technique_embeddings):
                            continue

                        tech_embedding = self.technique_embeddings[i]
                        similarity = self._cosine_similarity(
                            segment_embedding, tech_embedding
                        )

                        all_similarities.append((tech_id, similarity))

            similar_time = int((time.time() - similar_start) * 1000)

            # Record similarity computation performance
            if metrics_recorder and extractor_id:
                metrics_recorder.record_performance_benchmark(
                    extractor_name="optimized_semantic",
                    operation_type="similarity_computation",
                    execution_time_ms=similar_time,
                    input_size=len(self.technique_embeddings) * len(segment_embeddings),
                )

            # Aggregate similarities across segments
            technique_scores = {}

            for tech_id, similarity in all_similarities:
                if (
                    tech_id not in technique_scores
                    or similarity > technique_scores[tech_id]
                ):
                    technique_scores[tech_id] = similarity

                    # Record similarity if above threshold and metrics enabled
                    if metrics_recorder and extractor_id and similarity >= threshold:
                        # Find technique embedding
                        tech_idx = (
                            self.embedding_tech_ids.index(tech_id)
                            if tech_id in self.embedding_tech_ids
                            else -1
                        )
                        tech_embedding_dim = None
                        if tech_idx >= 0 and tech_idx < len(self.technique_embeddings):
                            tech_embedding_dim = len(
                                self.technique_embeddings[tech_idx]
                            )

                        # Record semantic score
                        metrics_recorder.record_semantic_scores(
                            extractor_id=extractor_id,
                            technique_id=tech_id,
                            similarity_score=similarity,
                            embedding_dimension=tech_embedding_dim,
                            model_used=self.model_name,
                        )

            # Filter by threshold
            filtered_similarities = [
                (tech_id, sim)
                for tech_id, sim in technique_scores.items()
                if sim >= threshold
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
                    "method": "optimized_semantic",
                }

                # Add technique name if available
                if tech_id in self.techniques_data:
                    result["name"] = self.techniques_data[tech_id].get("name", "")

                results.append(result)

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
