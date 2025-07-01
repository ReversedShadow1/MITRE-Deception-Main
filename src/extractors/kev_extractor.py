"""
KEV-based ATT&CK Technique Extractor
-----------------------------------
Identifies ATT&CK techniques from CVEs mentioned in text using KEV data.
"""

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.database.postgresql import get_db
from src.integrations.kev_mapper import KEVMapper

logger = logging.getLogger("KEVExtractor")


class KEVExtractor:
    """
    Extracts ATT&CK techniques from text based on CVE mentions and KEV data
    """

    def __init__(
        self, kev_mapper: KEVMapper, techniques_data: Dict, neo4j_connector=None
    ):
        """
        Initialize KEV extractor

        Args:
            kev_mapper: KEV mapper instance
            techniques_data: Dictionary of technique data
        """
        self.kev_mapper = kev_mapper
        self.techniques_data = techniques_data
        self.neo4j_connector = neo4j_connector

        logger.info("KEV extractor initialized")

    # Modify the extract_techniques method in src/extractors/kev_extractor.py
    def extract_techniques(
        self,
        text: str,
        threshold: float = 0.5,
        top_k: int = 10,
        job_id: str = None,
    ) -> List[Dict]:
        """
        Extract techniques from text based on CVE mentions with metrics recording

        Args:
            text: Input text
            threshold: Minimum confidence threshold
            top_k: Maximum number of results
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
            parameters = {"min_confidence": threshold, "max_results": top_k}
            extractor_id = metrics_recorder.record_extractor_result(
                extractor_name="kev",
                raw_input=text,
                raw_output={},  # Will update this later
                execution_time_ms=0,  # Will update this later
                parameters=parameters,
            )

        # Extract CVEs from text
        cves = self._extract_cves(text)

        if not cves:
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

        logger.info(f"Found {len(cves)} CVEs in text")

        # Record CVE extraction performance if metrics enabled
        if metrics_recorder and extractor_id:
            metrics_recorder.record_performance_benchmark(
                extractor_name="kev",
                operation_type="cve_extraction",
                execution_time_ms=int((time.time() - start_time) * 1000),
                input_size=len(text),
            )

        # Get techniques for each CVE
        all_techniques = []

        for cve_id in cves:
            # Find position of CVE in text for context
            cve_pos = text.find(cve_id)
            cve_context = ""

            if cve_pos >= 0:
                # Get surrounding context (50 chars before and after)
                start_pos = max(0, cve_pos - 50)
                end_pos = min(len(text), cve_pos + len(cve_id) + 50)
                cve_context = text[start_pos:end_pos]

            techniques = self.kev_mapper.get_techniques_for_cve(cve_id)

            # Record KEV details if metrics enabled
            if metrics_recorder and extractor_id:
                # Get entry date if available
                entry_date = None
                kev_entries = self.kev_mapper.get_kev_entries_for_cve(cve_id)
                if kev_entries and "dateAdded" in kev_entries[0]:
                    entry_date_str = kev_entries[0]["dateAdded"]
                    try:
                        entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d")
                    except:
                        pass

                # Record KEV details
                metrics_recorder.record_kev_details(
                    extractor_id=extractor_id,
                    cve_id=cve_id,
                    cve_mention_position=cve_pos if cve_pos >= 0 else None,
                    cve_mention_context=cve_context,
                    kev_entry_date=entry_date,
                    technique_mappings={
                        t["technique_id"]: t["confidence"] for t in techniques
                    },
                    confidence_scores={
                        t["technique_id"]: t["confidence"] for t in techniques
                    },
                )

            for technique in techniques:
                if technique.get("confidence", 0) >= threshold:
                    # Add method and CVE source
                    technique_entry = {
                        "technique_id": technique["technique_id"],
                        "confidence": technique["confidence"],
                        "method": "kev",
                        "cve_id": cve_id,
                        "source": technique.get("source", "cve_mapping"),
                    }

                    all_techniques.append(technique_entry)

        # Process scores to get technique results
        results = self._rank_and_filter_results(all_techniques, top_k)

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

    def _extract_cves(self, text: str) -> List[str]:
        """
        Extract CVE IDs from text

        Args:
            text: Input text

        Returns:
            List of CVE IDs
        """
        # More comprehensive CVE pattern
        cve_pattern = r"CVE-\d{4}-\d{1,7}"
        cves = re.findall(cve_pattern, text, re.IGNORECASE)

        # Normalize to uppercase
        normalized_cves = [cve.upper() for cve in cves]

        # Remove duplicates
        return list(set(normalized_cves))

    def _rank_and_filter_results(
        self, techniques: List[Dict], top_k: int
    ) -> List[Dict]:
        """
        Rank and filter technique results

        Args:
            techniques: List of technique matches
            top_k: Maximum number of results

        Returns:
            Filtered and ranked results
        """
        if not techniques:
            return []

        # Get valid technique IDs
        valid_technique_ids = set(self.techniques_data.keys())

        # Remove duplicates (same technique from different CVEs) and filter invalid IDs
        unique_techniques = {}

        for technique in techniques:
            tech_id = technique["technique_id"]

            # Skip invalid technique IDs
            if tech_id not in valid_technique_ids:
                continue

            if tech_id not in unique_techniques:
                unique_techniques[tech_id] = technique
            else:
                # Keep the higher confidence entry
                if technique["confidence"] > unique_techniques[tech_id]["confidence"]:
                    unique_techniques[tech_id] = technique

                # Add the CVE ID if it's different
                current_cve = unique_techniques[tech_id].get("cve_id", "")
                new_cve = technique.get("cve_id", "")

                if new_cve and new_cve != current_cve:
                    # Store as a list of CVEs
                    if "cve_ids" not in unique_techniques[tech_id]:
                        unique_techniques[tech_id]["cve_ids"] = [current_cve]

                    unique_techniques[tech_id]["cve_ids"].append(new_cve)

        # Convert to list
        result_list = list(unique_techniques.values())

        # Sort by confidence
        result_list.sort(key=lambda x: x["confidence"], reverse=True)

        # Limit to max_results
        return result_list[:top_k]
