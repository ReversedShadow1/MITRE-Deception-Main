# src/validation/quality_checks.py
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("QualityValidator")


class DataQualityValidator:
    """Validator for ensuring data quality throughout the system"""

    def __init__(self, techniques_data: Dict, technique_keywords: Dict):
        """Initialize the validator

        Args:
            techniques_data: Dictionary of technique data
            technique_keywords: Dictionary of technique keywords
        """
        self.techniques_data = techniques_data
        self.technique_keywords = technique_keywords
        # Initialize quality metrics
        self.quality_metrics = {
            "total_validations": 0,
            "total_issues": 0,
            "issue_count_by_type": {},
            "recent_validations": [],
        }

    def validate_schema(self, data: Dict, schema_type: str) -> List[str]:
        """
        Validate data against a predefined schema

        Args:
            data: Data to validate
            schema_type: Type of schema to validate against

        Returns:
            List of validation issues
        """
        issues = []

        # Define schema validation rules
        schemas = {
            "technique": {
                "required": ["technique_id", "name", "description"],
                "optional": [
                    "tactics",
                    "platforms",
                    "data_sources",
                    "detection",
                    "url",
                ],
                "types": {
                    "technique_id": str,
                    "name": str,
                    "description": str,
                    "tactics": list,
                    "platforms": list,
                    "data_sources": list,
                    "detection": str,
                    "url": str,
                },
            },
            "analysis_result": {
                "required": ["technique_id", "confidence", "method"],
                "optional": [
                    "matched_keywords",
                    "cve_id",
                    "name",
                    "description",
                    "tactics",
                ],
                "types": {
                    "technique_id": str,
                    "confidence": float,
                    "method": str,
                    "matched_keywords": list,
                    "cve_id": str,
                    "name": str,
                    "description": str,
                    "tactics": list,
                },
                "constraints": {
                    "confidence": lambda x: 0 <= x <= 1,
                    "technique_id": lambda x: bool(re.match(r"T\d+(?:\.\d+)?", x)),
                },
            },
        }

        # Get schema definition
        schema = schemas.get(schema_type)
        if not schema:
            issues.append(f"Unknown schema type: {schema_type}")
            return issues

        # Check required fields
        for field in schema["required"]:
            if field not in data:
                issues.append(f"Missing required field: {field}")

        # Check field types
        for field, expected_type in schema["types"].items():
            if field in data and not isinstance(data[field], expected_type):
                issues.append(
                    f"Field {field} has wrong type. Expected {expected_type.__name__}, got {type(data[field]).__name__}"
                )

        # Check constraints
        if "constraints" in schema:
            for field, constraint_func in schema["constraints"].items():
                if field in data and not constraint_func(data[field]):
                    issues.append(
                        f"Field {field} failed constraint check: {data[field]}"
                    )

        # Track validation metrics
        self._track_validation(schema_type, len(issues) > 0)

        return issues

    def validate_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        allowed_types: Dict[str, List[str]],
    ) -> List[str]:
        """
        Validate a relationship between two entities

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of relationship
            allowed_types: Dictionary mapping relationship types to allowed (source_type, target_type) pairs

        Returns:
            List of validation issues
        """
        issues = []

        # Check if relationship type is valid
        if relationship_type not in allowed_types:
            issues.append(f"Invalid relationship type: {relationship_type}")
            return issues

        # Extract entity types from IDs
        source_type = self._get_entity_type(source_id)
        target_type = self._get_entity_type(target_id)

        # Check if entity types are valid for this relationship
        valid_pairs = allowed_types[relationship_type]
        if (source_type, target_type) not in valid_pairs:
            valid_pairs_str = ", ".join([f"({s}->{t})" for s, t in valid_pairs])
            issues.append(
                f"Invalid entity types for relationship {relationship_type}: {source_type}->{target_type}. Allowed: {valid_pairs_str}"
            )

        return issues

    def _get_entity_type(self, entity_id: str) -> str:
        """Determine entity type from ID format"""
        if entity_id.startswith("T") and re.match(r"T\d+", entity_id):
            return "technique"
        elif entity_id.startswith("CVE-"):
            return "cve"
        elif entity_id.startswith("CWE-"):
            return "cwe"
        elif entity_id.startswith("CAPEC-"):
            return "capec"
        elif entity_id.startswith("S"):
            return "software"
        elif entity_id.startswith("G"):
            return "group"
        elif entity_id.startswith("M"):
            return "mitigation"
        else:
            return "unknown"

    def validate_temporal_coherence(self, update_sequence: List[Dict]) -> List[str]:
        """
        Validate the temporal coherence of a sequence of updates

        Args:
            update_sequence: List of update operations with timestamps

        Returns:
            List of validation issues
        """
        issues = []

        if not update_sequence:
            return issues

        # Sort by timestamp
        sorted_sequence = sorted(update_sequence, key=lambda x: x.get("timestamp", ""))

        # Check for out-of-order operations on the same entity
        entity_last_op = {}

        for update in sorted_sequence:
            entity_id = update.get("entity_id")
            operation = update.get("operation")
            timestamp = update.get("timestamp")

            if not entity_id or not operation or not timestamp:
                issues.append(f"Invalid update record: missing fields")
                continue

            if entity_id in entity_last_op:
                last_op = entity_last_op[entity_id]

                # Check for invalid operation sequences
                if last_op["operation"] == "delete" and operation != "create":
                    issues.append(
                        f"Invalid operation sequence for {entity_id}: {last_op['operation']} -> {operation}"
                    )

                # Check for missing versions
                if "version" in update and "version" in last_op:
                    if update["version"] > last_op["version"] + 1:
                        issues.append(
                            f"Missing versions for {entity_id}: {last_op['version']} -> {update['version']}"
                        )

            entity_last_op[entity_id] = update

        return issues

    def _track_validation(self, validation_type: str, has_issues: bool) -> None:
        """Track validation metrics"""
        self.quality_metrics["total_validations"] += 1

        if has_issues:
            self.quality_metrics["total_issues"] += 1

            if validation_type not in self.quality_metrics["issue_count_by_type"]:
                self.quality_metrics["issue_count_by_type"][validation_type] = 0

            self.quality_metrics["issue_count_by_type"][validation_type] += 1

        # Add to recent validations (keep last 100)
        self.quality_metrics["recent_validations"].append(
            {
                "type": validation_type,
                "has_issues": has_issues,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        if len(self.quality_metrics["recent_validations"]) > 100:
            self.quality_metrics["recent_validations"].pop(0)

    def get_quality_metrics(self) -> Dict:
        """Get current quality metrics"""
        # Calculate error rate
        if self.quality_metrics["total_validations"] > 0:
            error_rate = (
                self.quality_metrics["total_issues"]
                / self.quality_metrics["total_validations"]
            )
        else:
            error_rate = 0.0

        metrics = {
            **self.quality_metrics,
            "error_rate": error_rate,
            "report_time": datetime.utcnow().isoformat(),
        }

        return metrics

    def generate_quality_report(self) -> Dict:
        """Generate a comprehensive data quality report"""
        metrics = self.get_quality_metrics()

        # Add trending information
        if len(metrics["recent_validations"]) > 10:
            recent_error_count = sum(
                1 for v in metrics["recent_validations"][-10:] if v["has_issues"]
            )
            previous_error_count = sum(
                1 for v in metrics["recent_validations"][-20:-10] if v["has_issues"]
            )

            metrics["error_trend"] = (
                "improving"
                if recent_error_count < previous_error_count
                else "worsening"
                if recent_error_count > previous_error_count
                else "stable"
            )
        else:
            metrics["error_trend"] = "insufficient_data"

        return metrics

    def validate_technique_result(self, result: Dict) -> List[str]:
        """Validate a technique extraction result

        Args:
            result: Technique extraction result

        Returns:
            List of validation issues, empty if valid
        """
        issues = []

        # Check required fields
        required_fields = ["technique_id", "confidence", "method"]
        for field in required_fields:
            if field not in result:
                issues.append(f"Missing required field: {field}")

        if "technique_id" in result:
            # Validate technique ID format
            tech_id = result["technique_id"]
            if not re.match(r"T\d+(?:\.\d+)?", tech_id):
                issues.append(f"Invalid technique ID format: {tech_id}")

            # Check technique exists in known techniques
            if tech_id not in self.techniques_data:
                issues.append(f"Unknown technique ID: {tech_id}")

        # Validate confidence score
        if "confidence" in result:
            confidence = result["confidence"]
            if not isinstance(confidence, (int, float)):
                issues.append(f"Confidence must be numeric, got: {type(confidence)}")
            elif confidence < 0 or confidence > 1:
                issues.append(f"Confidence must be between 0 and 1, got: {confidence}")

        # Validate extraction method
        if "method" in result:
            valid_methods = [
                "rule_based",
                "bm25",
                "ner",
                "semantic",
                "classifier",
                "kev",
                "ensemble",
            ]
            method = result["method"]
            if method not in valid_methods and not any(
                m in method for m in valid_methods
            ):
                issues.append(f"Unknown extraction method: {method}")

        # Validate matched keywords if present
        if "matched_keywords" in result:
            keywords = result["matched_keywords"]
            if not isinstance(keywords, list):
                issues.append("matched_keywords must be a list")

        # Validate CVE ID if present
        if "cve_id" in result:
            cve_id = result["cve_id"]
            if not re.match(r"CVE-\d{4}-\d{1,7}", cve_id):
                issues.append(f"Invalid CVE ID format: {cve_id}")

        return issues

    def validate_extraction_results(self, results: List[Dict]) -> Dict[str, List[str]]:
        """Validate a list of extraction results

        Args:
            results: List of technique extraction results

        Returns:
            Dictionary mapping result index to list of issues
        """
        validation_issues = {}

        for i, result in enumerate(results):
            issues = self.validate_technique_result(result)
            if issues:
                validation_issues[i] = issues

        return validation_issues

    def validate_consistency(self, results: List[Dict]) -> List[str]:
        """Check for consistency issues across results

        Args:
            results: List of technique extraction results

        Returns:
            List of consistency issues
        """
        issues = []

        # Check for duplicated technique IDs
        tech_ids = {}
        for i, result in enumerate(results):
            tech_id = result.get("technique_id")
            if tech_id:
                if tech_id in tech_ids:
                    issues.append(
                        f"Duplicate technique ID {tech_id} at positions {tech_ids[tech_id]} and {i}"
                    )
                else:
                    tech_ids[tech_id] = i

        # Check for sub-techniques without parent techniques
        subtechs = set()
        parents = set()

        for result in results:
            tech_id = result.get("technique_id", "")
            if "." in tech_id:  # Sub-technique
                parent_id = tech_id.split(".")[0]
                subtechs.add(parent_id)
            else:
                parents.add(tech_id)

        orphaned_subtechs = subtechs - parents
        if orphaned_subtechs:
            issues.append(
                f"Sub-techniques present without parent techniques: {', '.join(orphaned_subtechs)}"
            )

        return issues
