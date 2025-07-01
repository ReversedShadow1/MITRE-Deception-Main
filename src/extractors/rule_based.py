"""
Rule-Based Extractor for ATT&CK Techniques
----------------------------------------
Implements keyword and rule-based matching for ATT&CK techniques.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set

from src.database.postgresql import get_db

logger = logging.getLogger("RuleBasedExtractor")


class RuleBasedExtractor:
    """
    Rule-based extractor for ATT&CK techniques using keyword matching
    """

    def __init__(
        self, technique_keywords: Dict, techniques_data: Dict, neo4j_connector=None
    ):
        """
        Initialize rule-based extractor

        Args:
            technique_keywords: Dictionary mapping technique IDs to keywords
            techniques_data: Dictionary of technique data
        """
        self.technique_keywords = technique_keywords
        self.techniques_data = techniques_data
        self.neo4j_connector = neo4j_connector

        # Build inverted index
        self.keyword_to_techniques = self._build_inverted_index()

        # Build regex patterns for faster matching
        self.technique_patterns = self._build_regex_patterns()

    def _build_inverted_index(self) -> Dict[str, List[str]]:
        """
        Build inverted index from keywords to techniques

        Returns:
            Dictionary mapping keywords to lists of technique IDs
        """
        index = {}

        for tech_id, keywords in self.technique_keywords.items():
            for keyword in keywords:
                if keyword not in index:
                    index[keyword] = []

                if tech_id not in index[keyword]:
                    index[keyword].append(tech_id)

        logger.info(f"Built inverted index with {len(index)} keywords")
        return index

    def _build_regex_patterns(self) -> Dict[str, re.Pattern]:
        """
        Build regex patterns for each technique's keywords

        Returns:
            Dictionary mapping technique IDs to compiled regex patterns
        """
        patterns = {}

        for tech_id, keywords in self.technique_keywords.items():
            if not keywords:
                continue

            # Sort keywords by length (longest first) to prioritize specific matches
            sorted_keywords = sorted(keywords, key=len, reverse=True)

            # Build regex pattern for this technique
            # Use word boundaries for more accurate matching
            pattern_str = (
                r"\b(?:" + "|".join(re.escape(kw) for kw in sorted_keywords) + r")\b"
            )

            try:
                patterns[tech_id] = re.compile(pattern_str, re.IGNORECASE)
            except re.error:
                logger.warning(
                    f"Failed to compile regex for {tech_id}, falling back to simple matching"
                )

        logger.info(f"Built {len(patterns)} regex patterns for techniques")
        return patterns

    def _check_additional_signals(self, text: str, matches: Dict) -> None:
        """
        Check for additional signals that might indicate specific techniques

        Args:
            text: Input text
            matches: Dictionary of current matches to update
        """
        text_lower = text.lower()

        # Check for indicators of specific techniques
        # These are common signals that strongly indicate certain techniques

        # T1190 - Exploit Public-Facing Application
        if any(
            term in text_lower
            for term in [
                "web exploit",
                "web application vulnerability",
                "public-facing",
            ]
        ):
            self._add_or_update_match(matches, "T1190", ["web exploit"], 2)

        # T1133 - External Remote Services
        if any(
            term in text_lower for term in ["vpn", "rdp exploit", "external service"]
        ):
            self._add_or_update_match(matches, "T1133", ["vpn", "remote service"], 2)

        # T1566 - Phishing
        if any(
            term in text_lower
            for term in ["phishing", "spear phishing", "malicious email"]
        ):
            self._add_or_update_match(matches, "T1566", ["phishing"], 2)

        # T1059 - Command and Scripting Interpreter
        if any(
            term in text_lower
            for term in ["powershell", "cmd.exe", "bash", "python script"]
        ):
            self._add_or_update_match(
                matches, "T1059", ["script", "command interpreter"], 2
            )

        # T1027 - Obfuscated Files or Information
        if any(
            term in text_lower
            for term in ["obfuscated", "encoded", "encrypted payload"]
        ):
            self._add_or_update_match(matches, "T1027", ["obfuscated", "encoded"], 2)

    def _add_or_update_match(
        self, matches: Dict, tech_id: str, keywords: List[str], count: int
    ) -> None:
        """
        Add or update a match in the matches dictionary

        Args:
            matches: Dictionary of current matches
            tech_id: Technique ID to add/update
            keywords: List of keywords that matched
            count: Number of matches to add
        """
        if tech_id not in matches:
            matches[tech_id] = {"count": count, "keywords": keywords}
        else:
            # Update existing match
            matches[tech_id]["count"] += count

            # Add new keywords
            for keyword in keywords:
                if keyword not in matches[tech_id]["keywords"]:
                    matches[tech_id]["keywords"].append(keyword)

    def _score_matches(self, matches: Dict) -> List[Dict]:
        """
        Convert matches to results with confidence scores

        Args:
            matches: Dictionary of technique matches

        Returns:
            List of results with confidence scores
        """
        results = []

        for tech_id, match_data in matches.items():
            # Get match data
            match_count = match_data["count"]
            keywords = match_data["keywords"]
            unique_keywords = len(set(keywords))

            # Calculate base confidence based on number of unique keywords
            # More unique keywords = higher confidence
            base_confidence = min(0.4 + (unique_keywords * 0.1), 0.9)

            # Create result
            result = {
                "technique_id": tech_id,
                "confidence": base_confidence,
                "match_count": match_count,
                "matched_keywords": list(set(keywords)),
                "method": "rule_based",
            }

            results.append(result)

        return results

    def _check_contextual_patterns(self, text: str, matches: Dict) -> None:
        """
        Check for contextual patterns that imply specific techniques
        even when technique names aren't explicitly mentioned

        Args:
            text: Input text
            matches: Dictionary of current matches to update
        """
        text_lower = text.lower()

        # Contextual patterns for command and control
        if (
            re.search(r"(outbound|external)\s+connection", text_lower)
            or re.search(r"(command|control)\s+server", text_lower)
            or re.search(r"beacon\s+(to|back)", text_lower)
        ):
            self._add_or_update_match(
                matches, "T1071", ["command and control", "external communication"], 2
            )

        # Contextual patterns for privilege escalation
        if re.search(r"(elevated|higher|admin)\s+privilege", text_lower) or re.search(
            r"(escalate|increase)\s+(privilege|permission)", text_lower
        ):
            self._add_or_update_match(
                matches, "T1068", ["privilege escalation", "elevation"], 2
            )

        # Contextual patterns for lateral movement
        if re.search(
            r"(move|spread|pivot)\s+(across|between|to other)\s+(system|host|machine|network)",
            text_lower,
        ) or re.search(r"lateral\s+movement", text_lower):
            self._add_or_update_match(
                matches, "T1021", ["lateral movement", "remote services"], 2
            )

        # Contextual patterns for persistence
        if re.search(
            r"(maintain|establish|ensure)\s+(access|presence)", text_lower
        ) or re.search(r"(persist|persistent|persistence)", text_lower):
            self._add_or_update_match(
                matches, "T1053", ["persistence", "scheduled task"], 2
            )

        # Contextual patterns for defense evasion
        if re.search(
            r"(avoid|evade|bypass)\s+(detection|monitoring|defense)", text_lower
        ) or re.search(r"(disable|impair)\s+(security|antivirus|firewall)", text_lower):
            self._add_or_update_match(
                matches, "T1562", ["defense evasion", "impair defenses"], 2
            )

    # Modify the extract_techniques method in src/extractors/rule_based.py
    def extract_techniques(
        self,
        text: str,
        min_confidence: float = 0.1,
        max_results: int = 10,
        job_id: str = None,
    ) -> List[Dict]:
        """
        Extract techniques using rule-based matching with metrics recording

        Args:
            text: Input text to analyze
            min_confidence: Minimum confidence threshold
            max_results: Maximum number of results to return
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

        # Normalize text
        text_lower = text.lower()

        # Track matches for each technique
        matches = {}

        # Record extractor execution details if metrics enabled
        extractor_id = None
        if metrics_recorder:
            parameters = {"min_confidence": min_confidence, "max_results": max_results}
            extractor_id = metrics_recorder.record_extractor_result(
                extractor_name="rule_based",
                raw_input=text,
                raw_output={},  # Will update this later
                execution_time_ms=0,  # Will update this later
                parameters=parameters,
            )

        # Method 1: Use regex patterns for direct matching
        for tech_id, pattern in self.technique_patterns.items():
            # Find all matches
            all_matches = pattern.findall(text)

            if all_matches:
                # Count unique matched keywords
                unique_keywords = set(match.lower() for match in all_matches)
                total_matches = len(all_matches)

                # Record keyword matches if metrics enabled
                if metrics_recorder and extractor_id:
                    keyword_data = []
                    for i, match in enumerate(all_matches):
                        # Get match position if possible
                        match_pos = text_lower.find(match.lower())
                        # Get surrounding context
                        start_pos = max(0, match_pos - 50)
                        end_pos = min(len(text), match_pos + len(match) + 50)
                        context = text[start_pos:end_pos]

                        keyword_data.append(
                            {"text": match, "position": match_pos, "context": context}
                        )

                    metrics_recorder.record_keywords(
                        extractor_id=extractor_id,
                        technique_id=tech_id,
                        keywords=keyword_data,
                    )

                if tech_id not in matches:
                    matches[tech_id] = {
                        "count": total_matches,
                        "keywords": list(unique_keywords),
                    }
                else:
                    # Update existing match
                    matches[tech_id]["count"] += total_matches
                    matches[tech_id]["keywords"].extend(list(unique_keywords))

        # Method 2: Check for specific phrases and signals
        self._check_additional_signals(text, matches)

        # Convert matches to results with confidence scoring
        results = self._score_matches(matches)

        # Filter by confidence threshold
        results = [r for r in results if r["confidence"] >= min_confidence]

        # Sort by confidence (descending) and limit results
        results.sort(key=lambda x: x["confidence"], reverse=True)
        results = results[:max_results]

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

            get_db().execute(query, (Json(results), execution_time_ms, extractor_id))

        return results
