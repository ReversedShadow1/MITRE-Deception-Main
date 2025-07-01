"""
Advanced Ensemble Method for ATT&CK Techniques
--------------------------------------------
Implements an adaptive ensemble with context-aware weighting and learning capabilities.
"""

import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from src.database.postgresql import get_db

logger = logging.getLogger("AdvancedEnsemble")


class AdvancedEnsembleMethod:
    """
    Advanced ensemble method for combining results from multiple extractors
    with adaptive weighting, contextual boosting, and learning capabilities
    """

    def __init__(
        self,
        techniques_data: Dict,
        technique_keywords: Dict,
        models_dir: str = "models/ensemble",
        use_calibration: bool = True,
        use_adaptive_weights: bool = True,
        use_neo4j: bool = True,
        neo4j_connector=None,
    ):
        """
        Initialize advanced ensemble method

        Args:
            techniques_data: Dictionary of technique data
            technique_keywords: Dictionary of technique keywords
            models_dir: Directory for model storage
            use_calibration: Whether to use confidence calibration
            use_adaptive_weights: Whether to use adaptive weights
            use_neo4j: Whether to use Neo4j for relationship-based boosting
            neo4j_connector: Optional Neo4j connector
        """
        self.techniques_data = techniques_data
        self.technique_keywords = technique_keywords
        self.models_dir = models_dir
        self.use_calibration = use_calibration
        self.use_adaptive_weights = use_adaptive_weights
        self.use_neo4j = use_neo4j
        self.neo4j_connector = neo4j_connector

        # Create models directory
        os.makedirs(models_dir, exist_ok=True)

        # Base weights for different extractors
        self.base_weights = {
            "enhanced_rule_based": 0.25,
            "enhanced_bm25": 0.20,
            "optimized_ner": 0.20,
            "optimized_semantic": 0.15,
            "kev": 0.30,  # Higher weight for KEV (high precision)
            "rule_based": 0.20,  # Fallback for original rule-based
            "bm25": 0.15,  # Fallback for original BM25
            "ner": 0.15,  # Fallback for original NER
            "semantic": 0.10,  # Fallback for original semantic
            "classifier": 0.10,  # Fallback for original classifier
        }

        # Path for learned weights
        self.weights_path = os.path.join(models_dir, "learned_weights.json")

        # Load learned weights if available
        self.learned_weights = self._load_learned_weights()

        # Technique relationship cache
        self.relationship_cache = {}

        # Confidence calibration parameters
        self.calibration_params = {
            "enhanced_rule_based": {"scale": 0.9, "shift": 0.0},
            "enhanced_bm25": {"scale": 0.8, "shift": 0.05},
            "optimized_ner": {"scale": 0.7, "shift": 0.1},
            "optimized_semantic": {"scale": 0.8, "shift": 0.05},
            "kev": {"scale": 1.0, "shift": 0.0},  # KEV is already well-calibrated
            "rule_based": {"scale": 0.8, "shift": 0.0},
            "bm25": {"scale": 0.7, "shift": 0.05},
            "ner": {"scale": 0.6, "shift": 0.1},
            "semantic": {"scale": 0.7, "shift": 0.05},
            "classifier": {"scale": 0.8, "shift": 0.0},
        }

    def _load_learned_weights(self) -> Dict:
        """
        Load learned weights from file

        Returns:
            Dictionary of learned weights
        """
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load learned weights: {e}")

        return {}

    def _save_learned_weights(self) -> None:
        """Save learned weights to file"""
        try:
            with open(self.weights_path, "w") as f:
                json.dump(self.learned_weights, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learned weights: {e}")

    def _get_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze text complexity to determine optimal extractor weights

        Args:
            text: Input text

        Returns:
            Dictionary of complexity metrics
        """
        metrics = {}

        # Length metrics
        metrics["char_count"] = len(text)
        metrics["word_count"] = len(re.findall(r"\b\w+\b", text))

        # Get sentence count
        sentences = re.split(r"[.!?]+", text)
        metrics["sentence_count"] = len([s for s in sentences if s.strip()])

        # Calculate average sentence length
        if metrics["sentence_count"] > 0:
            metrics["avg_sentence_length"] = (
                metrics["word_count"] / metrics["sentence_count"]
            )
        else:
            metrics["avg_sentence_length"] = 0

        # Check for technical content
        tech_patterns = [
            r"\b(?:CVE-\d{4}-\d{1,7})\b",  # CVEs
            r"\b(?:RFC\s?\d{3,4})\b",  # RFCs
            r"\b(?:IPv[46]|TCP|UDP|HTTP[S]?|FTP|SSH|TLS|SSL)\b",  # Protocols
            r"\b(?:0x[0-9a-f]+)\b",  # Hex values
            r"\b(?:[a-f0-9]{32}|[a-f0-9]{40}|[a-f0-9]{64})\b",  # Hashes
            r"(?:\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)",  # IP addresses
        ]

        # Count technical pattern matches
        tech_matches = sum(
            len(re.findall(pattern, text, re.IGNORECASE)) for pattern in tech_patterns
        )
        metrics["technical_density"] = tech_matches / max(metrics["word_count"], 1)

        # Count cybersecurity terms
        security_terms = [
            r"\b(?:malware|ransomware|virus|trojan|worm|botnet|backdoor|rootkit|keylogger|spyware)\b",
            r"\b(?:exploit|vulnerability|patch|zero-day|attack|threat|compromise|breach|incident)\b",
            r"\b(?:authentication|authorization|encryption|decryption|hash|cipher|key|credential)\b",
            r"\b(?:firewall|IDS|IPS|SIEM|EDR|XDR|NDR|SOAR|SOC|MSSP)\b",
        ]

        security_matches = sum(
            len(re.findall(pattern, text, re.IGNORECASE)) for pattern in security_terms
        )
        metrics["security_term_density"] = security_matches / max(
            metrics["word_count"], 1
        )

        # Calculate overall technical score (0-1)
        metrics["technical_score"] = min(
            1.0,
            (
                0.3 * min(1.0, metrics["technical_density"] * 10)
                + 0.7 * min(1.0, metrics["security_term_density"] * 5)
            ),
        )

        return metrics

    def _adapt_weights(self, text: str) -> Dict[str, float]:
        """
        Adapt extractor weights based on text characteristics

        Args:
            text: Input text

        Returns:
            Dictionary of adapted weights
        """
        # Start with base weights
        weights = self.base_weights.copy()

        # Get text complexity metrics
        metrics = self._get_text_complexity(text)

        # Adjust weights based on text characteristics
        if metrics["technical_score"] > 0.7:
            # Highly technical text - boost NER and BM25
            weights["enhanced_bm25"] *= 1.2
            weights["optimized_ner"] *= 1.2
            weights["enhanced_rule_based"] *= 0.9
        elif metrics["technical_score"] < 0.3:
            # Less technical text - boost rule-based and semantic
            weights["enhanced_rule_based"] *= 1.2
            weights["optimized_semantic"] *= 1.2
            weights["enhanced_bm25"] *= 0.9

        # Adjust for text length
        if metrics["word_count"] > 1000:
            # Long text - semantic becomes more important
            weights["optimized_semantic"] *= 1.2
            weights["enhanced_rule_based"] *= 0.9
        elif metrics["word_count"] < 100:
            # Short text - rule-based and BM25 more reliable
            weights["enhanced_rule_based"] *= 1.2
            weights["enhanced_bm25"] *= 1.1
            weights["optimized_semantic"] *= 0.8

        # Adjust for sentence complexity
        if metrics["avg_sentence_length"] > 25:
            # Complex sentences - NER and semantic more useful
            weights["optimized_ner"] *= 1.1
            weights["optimized_semantic"] *= 1.1
            weights["enhanced_rule_based"] *= 0.9

        # Always keep KEV at high weight
        weights["kev"] = max(weights["kev"], self.base_weights["kev"])

        # Normalize weights
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _calibrate_confidence(self, method: str, confidence: float) -> float:
        """
        Apply calibration to confidence scores

        Args:
            method: Extraction method
            confidence: Raw confidence score

        Returns:
            Calibrated confidence score
        """
        if not self.use_calibration:
            return confidence

        # Get calibration parameters
        params = self.calibration_params.get(method, {"scale": 0.8, "shift": 0.0})

        # Apply linear transformation
        calibrated = confidence * params["scale"] + params["shift"]

        # Ensure valid range
        return max(0.0, min(1.0, calibrated))

    def _get_related_techniques(self, technique_id: str, depth: int = 1) -> Set[str]:
        """
        Get related techniques using Neo4j

        Args:
            technique_id: Technique ID
            depth: Relationship depth to explore

        Returns:
            Set of related technique IDs
        """
        # Check cache first
        cache_key = f"{technique_id}_{depth}"
        if cache_key in self.relationship_cache:
            return self.relationship_cache[cache_key]

        related_techniques = set()

        # Use Neo4j if available
        if self.use_neo4j and self.neo4j_connector:
            try:
                # Query for related techniques
                query = """
                MATCH (t:AttackTechnique {technique_id: $technique_id})-[r:RELATED_TO|SUBTECHNIQUE_OF|PARENT_OF|SIMILAR_TO*1..2]-(related:AttackTechnique)
                RETURN related.technique_id as related_id
                """

                results = self.neo4j_connector.run_query(
                    query, {"technique_id": technique_id}
                )

                # Extract technique IDs
                for result in results:
                    if "related_id" in result:
                        related_techniques.add(result["related_id"])

            except Exception as e:
                logger.error(f"Error getting related techniques: {e}")

        # Add to cache
        self.relationship_cache[cache_key] = related_techniques

        return related_techniques

    def ensemble_extractors(
        self,
        text: str,
        extractor_results: Dict[str, List[Dict]],
        threshold: float = 0.2,
        max_results: int = 10,
        job_id: str = None,
    ) -> List[Dict]:
        """
        Combine results from multiple extractors with adaptive ensemble and metrics recording

        Args:
            text: Input text
            extractor_results: Dictionary mapping extractor names to result lists
            threshold: Minimum confidence threshold
            max_results: Maximum number of results
            job_id: Optional job ID for metrics recording

        Returns:
            List of ensemble results with confidence scores
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
                "max_results": max_results,
                "extractor_count": len(extractor_results),
                "extractors": list(extractor_results.keys()),
                "use_calibration": self.use_calibration,
                "use_adaptive_weights": self.use_adaptive_weights,
                "use_neo4j": self.use_neo4j,
            }
            extractor_id = metrics_recorder.record_extractor_result(
                extractor_name="advanced_ensemble",
                raw_input=text,
                raw_output={},  # Will update this later
                execution_time_ms=0,  # Will update this later
                parameters=parameters,
            )

        # Get adaptive weights if enabled
        weight_start = time.time()
        if self.use_adaptive_weights:
            weights = self._adapt_weights(text)
        else:
            weights = self.base_weights
        weight_time = int((time.time() - weight_start) * 1000)

        # Record weight adaptation performance
        if metrics_recorder and extractor_id:
            metrics_recorder.record_performance_benchmark(
                extractor_name="advanced_ensemble",
                operation_type="weight_adaptation",
                execution_time_ms=weight_time,
                input_size=len(text),
            )

        logger.info(f"Using ensemble weights: {weights}")

        # Get all unique technique IDs from results
        technique_ids = set()
        for extractor_name, results in extractor_results.items():
            for result in results:
                if "technique_id" in result:
                    technique_ids.add(result["technique_id"])

        # Process each technique
        ensemble_results = []

        for tech_id in technique_ids:
            # Track scores, methods, and features for this technique
            tech_scores = {}
            matched_keywords = set()
            matched_entities = set()
            entity_types = set()
            matched_cves = set()
            component_scores = {}

            # Collect scores from each extractor
            for extractor_name, results in extractor_results.items():
                # Find technique in this extractor's results
                for result in results:
                    if result.get("technique_id") == tech_id:
                        # Get confidence score
                        raw_confidence = result.get("confidence", 0.0)

                        # Apply calibration
                        if self.use_calibration:
                            calibrated_confidence = self._calibrate_confidence(
                                extractor_name, raw_confidence
                            )
                        else:
                            calibrated_confidence = raw_confidence

                        # Store score
                        tech_scores[extractor_name] = calibrated_confidence
                        component_scores[extractor_name] = raw_confidence

                        # Collect matched keywords
                        if "matched_keywords" in result:
                            matched_keywords.update(result["matched_keywords"])

                        # Collect matched entities
                        if "matched_entities" in result:
                            matched_entities.update(result["matched_entities"])

                        # Collect entity types
                        if "entity_types" in result:
                            entity_types.update(result["entity_types"])

                        # Collect matched CVEs
                        if "cve_id" in result:
                            matched_cves.add(result["cve_id"])

                        # Only use first match for this extractor
                        break

            # Skip if no scores
            if not tech_scores:
                continue

            # Calculate weighted sum of scores
            weighted_sum = 0.0
            weight_sum = 0.0

            for extractor_name, score in tech_scores.items():
                extractor_weight = weights.get(extractor_name, 0.1)
                weighted_sum += score * extractor_weight
                weight_sum += extractor_weight

            # Normalize by sum of weights
            if weight_sum > 0:
                base_confidence = weighted_sum / weight_sum
            else:
                base_confidence = 0.0

            # Apply confirmation boost when multiple extractors agree
            extractor_count = len(tech_scores)

            if extractor_count > 1:
                # Higher boost for more agreeing extractors
                diversity_boost = 1.0 + (0.1 * min(extractor_count - 1, 3))
                confidence = min(base_confidence * diversity_boost, 0.98)
            else:
                confidence = base_confidence

            # Apply Neo4j relationship boost if enabled
            if self.use_neo4j and extractor_count >= 2:
                neo4j_start = time.time()
                # Get techniques detected by other extractors
                other_techniques = set()
                for extractor_name, results in extractor_results.items():
                    for result in results:
                        other_tech_id = result.get("technique_id")
                        if other_tech_id and other_tech_id != tech_id:
                            other_techniques.add(other_tech_id)

                # Get related techniques for this one
                related_techniques = self._get_related_techniques(tech_id)

                # Check overlap with other detected techniques
                related_overlap = related_techniques.intersection(other_techniques)

                if related_overlap:
                    # Apply relationship boost
                    relationship_boost = 1.0 + (0.05 * min(len(related_overlap), 3))
                    confidence = min(confidence * relationship_boost, 0.98)

                neo4j_time = int((time.time() - neo4j_start) * 1000)

                # Record Neo4j relationship boost performance
                if metrics_recorder and extractor_id:
                    metrics_recorder.record_performance_benchmark(
                        extractor_name="advanced_ensemble",
                        operation_type="neo4j_relationship_boost",
                        execution_time_ms=neo4j_time,
                        input_size=len(other_techniques),
                    )

            # Skip low confidence results
            if confidence < threshold:
                continue

            # Create ensemble result
            ensemble_result = {
                "technique_id": tech_id,
                "confidence": confidence,
                "method": "advanced_ensemble",
                "component_scores": component_scores,
                "extractors_used": list(tech_scores.keys()),
            }

            # Add technique name if available
            if tech_id in self.techniques_data:
                ensemble_result["name"] = self.techniques_data[tech_id].get("name", "")

                # Add technique description (truncated)
                description = self.techniques_data[tech_id].get("description", "")
                if description:
                    ensemble_result["description"] = (
                        description[:200] + "..."
                        if len(description) > 200
                        else description
                    )

            # Add matched features
            if matched_keywords:
                ensemble_result["matched_keywords"] = list(matched_keywords)

            if matched_entities:
                ensemble_result["matched_entities"] = list(matched_entities)

            if entity_types:
                ensemble_result["entity_types"] = list(entity_types)

            if matched_cves:
                if len(matched_cves) == 1:
                    ensemble_result["cve_id"] = list(matched_cves)[0]
                else:
                    ensemble_result["cve_ids"] = list(matched_cves)

            # Record ensemble details if metrics enabled
            if metrics_recorder:
                metrics_recorder.record_ensemble_details(
                    technique_id=tech_id,
                    ensemble_method="advanced_ensemble",
                    final_confidence=confidence,
                    component_scores=component_scores,
                    weights_used={k: v for k, v in weights.items() if k in tech_scores},
                )

            ensemble_results.append(ensemble_result)

        # Sort by confidence and limit results
        ensemble_results.sort(key=lambda x: x["confidence"], reverse=True)
        final_results = ensemble_results[:max_results]

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
                query, (Json(final_results), execution_time_ms, extractor_id)
            )

        return final_results

    def update_from_feedback(self, feedback_data: Dict) -> None:
        """
        Update ensemble parameters based on user feedback

        Args:
            feedback_data: Dictionary with feedback data
        """
        try:
            # Extract key information from feedback
            feedback_type = feedback_data.get("feedback_type")
            technique_id = feedback_data.get("technique_id")
            suggested_technique = feedback_data.get("suggested_technique_id")

            if not feedback_type or not technique_id:
                logger.error("Invalid feedback data: missing required fields")
                return

            # Handle different feedback types
            if feedback_type == "correct":
                # Correct technique was identified - reinforce extractor weights
                if "extractors_used" in feedback_data:
                    for extractor in feedback_data["extractors_used"]:
                        # Get current learned weight
                        current_weight = self.learned_weights.get(extractor, 1.0)
                        # Slightly increase weight
                        self.learned_weights[extractor] = min(
                            current_weight * 1.05, 1.5
                        )

            elif feedback_type == "incorrect" and suggested_technique:
                # Incorrect technique was identified - adjust extractor weights
                if "extractors_used" in feedback_data:
                    for extractor in feedback_data["extractors_used"]:
                        # Get current learned weight
                        current_weight = self.learned_weights.get(extractor, 1.0)
                        # Slightly decrease weight
                        self.learned_weights[extractor] = max(
                            current_weight * 0.95, 0.5
                        )

            # Save updated weights
            self._save_learned_weights()

        except Exception as e:
            logger.error(f"Error updating from feedback: {e}")
