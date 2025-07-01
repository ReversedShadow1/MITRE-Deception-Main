"""
Enhanced Rule-Based Extractor for ATT&CK Techniques
-------------------------------------------------
Implements Aho-Corasick algorithm for efficient multi-pattern matching of ATT&CK techniques.
"""

import logging
import re
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.database.postgresql import get_db

logger = logging.getLogger("EnhancedRuleBasedExtractor")


class AhoCorasickAutomaton:
    """Efficient implementation of Aho-Corasick algorithm for multi-pattern matching"""

    def __init__(self):
        """Initialize the automaton with an empty trie"""
        self.trie = {}
        self.outputs = {}
        self.fail = {}
        self.built = False

    def add_pattern(self, pattern: str, pattern_id: str) -> None:
        """
        Add a pattern to the automaton

        Args:
            pattern: The pattern string to add
            pattern_id: The identifier for this pattern
        """
        current_state = 0

        # Ensure the trie root exists
        if 0 not in self.trie:
            self.trie[0] = {}

        # Add pattern to trie
        for char in pattern.lower():
            if current_state in self.trie and char in self.trie[current_state]:
                current_state = self.trie[current_state][char]
            else:
                if current_state not in self.trie:
                    self.trie[current_state] = {}

                next_state = len(self.trie)
                self.trie[current_state][char] = next_state

                if next_state not in self.trie:
                    self.trie[next_state] = {}

                current_state = next_state

        # Add output for this pattern
        if current_state not in self.outputs:
            self.outputs[current_state] = []

        self.outputs[current_state].append((pattern, pattern_id))
        self.built = False  # Require rebuilding after adding patterns

    def build_failure_function(self) -> None:
        """Build the failure function for the automaton"""
        queue = deque()

        # Initialize failure function for depth 1 nodes
        for char, state in self.trie[0].items():
            self.fail[state] = 0
            queue.append(state)

        # Build failure function for the rest
        while queue:
            current_state = queue.popleft()

            for char, next_state in self.trie.get(current_state, {}).items():
                queue.append(next_state)

                # Start from failure state of current
                failure_state = self.fail[current_state]

                # Find appropriate failure state
                while failure_state != 0 and char not in self.trie.get(
                    failure_state, {}
                ):
                    failure_state = self.fail[failure_state]

                if char in self.trie.get(failure_state, {}):
                    failure_state = self.trie[failure_state][char]

                self.fail[next_state] = failure_state

                # Add outputs from failure state
                if next_state not in self.outputs:
                    self.outputs[next_state] = []

                if failure_state in self.outputs:
                    self.outputs[next_state].extend(self.outputs[failure_state])

        self.built = True

    def search(self, text: str) -> Dict[str, List[Tuple[str, int]]]:
        """
        Search for all patterns in the text

        Args:
            text: The text to search in

        Returns:
            Dictionary mapping pattern_id to a list of (matched_pattern, position) tuples
        """
        if not self.built:
            self.build_failure_function()

        matches = defaultdict(list)
        text = text.lower()
        current_state = 0

        for i, char in enumerate(text):
            # Find next state
            while current_state != 0 and char not in self.trie.get(current_state, {}):
                current_state = self.fail[current_state]

            if char in self.trie.get(current_state, {}):
                current_state = self.trie[current_state][char]

            # Collect outputs (matches)
            if current_state in self.outputs:
                for pattern, pattern_id in self.outputs[current_state]:
                    match_position = i - len(pattern) + 1
                    matches[pattern_id].append((pattern, match_position))

        return dict(matches)


class EnhancedRuleBasedExtractor:
    """
    Enhanced rule-based extractor for ATT&CK techniques using Aho-Corasick algorithm
    and advanced contextual pattern matching
    """

    def __init__(
        self,
        technique_keywords: Dict,
        techniques_data: Dict,
        neo4j_connector=None,
        use_aho_corasick: bool = True,
        use_contextual_boost: bool = True,
    ):
        """
        Initialize the enhanced rule-based extractor

        Args:
            technique_keywords: Dictionary mapping technique IDs to keywords
            techniques_data: Dictionary of technique data
            neo4j_connector: Optional Neo4j connector for enhanced context
            use_aho_corasick: Whether to use Aho-Corasick algorithm (otherwise fall back to regex)
            use_contextual_boost: Whether to use contextual boosting for confidence scores
        """
        self.technique_keywords = technique_keywords
        self.techniques_data = techniques_data
        self.neo4j_connector = neo4j_connector
        self.use_aho_corasick = use_aho_corasick
        self.use_contextual_boost = use_contextual_boost

        # Build either Aho-Corasick automaton or traditional regex patterns
        if self.use_aho_corasick:
            self.automaton = self._build_automaton()
            self.technique_patterns = None
            logger.info(
                f"Built Aho-Corasick automaton for efficient multi-pattern matching"
            )
        else:
            self.automaton = None
            self.technique_patterns = self._build_regex_patterns()
            logger.info(
                f"Built {len(self.technique_patterns)} regex patterns for techniques"
            )

        # Precompute keyword-to-technique mapping for O(1) lookups
        self.keyword_to_techniques = self._build_keyword_mapping()

        # Security-focused context patterns
        self.context_patterns = self._build_context_patterns()

    def _build_automaton(self) -> AhoCorasickAutomaton:
        """
        Build Aho-Corasick automaton from technique keywords

        Returns:
            Initialized AhoCorasickAutomaton
        """
        automaton = AhoCorasickAutomaton()
        keyword_count = 0

        # Add all keywords to the automaton
        for tech_id, keywords in self.technique_keywords.items():
            if not keywords:
                continue

            for keyword in keywords:
                if len(keyword) > 2:  # Skip very short keywords
                    automaton.add_pattern(keyword, tech_id)
                    keyword_count += 1

        # Build the failure function
        automaton.build_failure_function()
        logger.info(f"Added {keyword_count} keywords to Aho-Corasick automaton")

        return automaton

    def _build_regex_patterns(self) -> Dict[str, re.Pattern]:
        """
        Build optimized regex patterns as fallback

        Returns:
            Dictionary mapping technique IDs to compiled regex patterns
        """
        patterns = {}

        # Group keywords by technique for batch regex compilation
        for tech_id, keywords in self.technique_keywords.items():
            if not keywords:
                continue

            # Sort keywords by length (longest first) to prioritize specific matches
            sorted_keywords = sorted(keywords, key=len, reverse=True)

            # Build optimized regex pattern with word boundaries
            pattern_str = (
                r"\b(?:" + "|".join(re.escape(kw) for kw in sorted_keywords) + r")\b"
            )

            try:
                patterns[tech_id] = re.compile(pattern_str, re.IGNORECASE)
            except re.error:
                logger.warning(
                    f"Failed to compile regex for {tech_id}, falling back to simple matching"
                )

        return patterns

    def _build_keyword_mapping(self) -> Dict[str, List[str]]:
        """
        Build inverted index from keywords to techniques for O(1) lookups

        Returns:
            Dictionary mapping lowercase keywords to lists of technique IDs
        """
        index = {}

        for tech_id, keywords in self.technique_keywords.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()

                if keyword_lower not in index:
                    index[keyword_lower] = []

                if tech_id not in index[keyword_lower]:
                    index[keyword_lower].append(tech_id)

        return index

    def _build_context_patterns(self) -> Dict[str, Dict]:
        """
        Build patterns for contextual matching to enhance detection accuracy

        Returns:
            Dictionary of context patterns by category
        """
        # Tactical patterns aligned with ATT&CK tactics
        patterns = {
            "initial_access": {
                "pattern": re.compile(
                    r"\b(phish|spearphish|malicious\s+(?:email|attachment)|exploit\s+(?:web|application|public))",
                    re.IGNORECASE,
                ),
                "techniques": ["T1566", "T1566.001", "T1566.002", "T1190"],
            },
            "execution": {
                "pattern": re.compile(
                    r"\b(execut(?:e|ion|ed)|script|command|run\s+(?:code|process))",
                    re.IGNORECASE,
                ),
                "techniques": ["T1059", "T1059.001", "T1059.003", "T1106"],
            },
            "persistence": {
                "pattern": re.compile(
                    r"\b(persist(?:ence|ent)|startup|boot|registry|schedule|cron|autorun)",
                    re.IGNORECASE,
                ),
                "techniques": ["T1053", "T1053.005", "T1547", "T1547.001"],
            },
            "privilege_escalation": {
                "pattern": re.compile(
                    r"\b(privile(?:ge|ged)|escalat(?:e|ion)|admin|root|sudo|uac)",
                    re.IGNORECASE,
                ),
                "techniques": ["T1068", "T1548", "T1548.002"],
            },
            "defense_evasion": {
                "pattern": re.compile(
                    r"\b(evade|evasion|bypass|disable|tamper|anti[\s-]?virus|detection)",
                    re.IGNORECASE,
                ),
                "techniques": ["T1562", "T1562.001", "T1070"],
            },
            "credential_access": {
                "pattern": re.compile(
                    r"\b(credential|password|hash|kerberos|ticket|token|dump)",
                    re.IGNORECASE,
                ),
                "techniques": ["T1003", "T1003.001", "T1110", "T1110.002"],
            },
            "discovery": {
                "pattern": re.compile(
                    r"\b(discover|enumerat(?:e|ion)|scan|query|list|network)",
                    re.IGNORECASE,
                ),
                "techniques": ["T1046", "T1082", "T1018"],
            },
            "lateral_movement": {
                "pattern": re.compile(
                    r"\b(lateral|move(?:ment)?|pivot|remote|psexec|wmi|smb)",
                    re.IGNORECASE,
                ),
                "techniques": ["T1021", "T1021.002", "T1021.006", "T1091"],
            },
            "collection": {
                "pattern": re.compile(
                    r"\b(collect|gather|harvest|screenshot|keylog|clipboard)",
                    re.IGNORECASE,
                ),
                "techniques": ["T1056", "T1056.001", "T1113", "T1115"],
            },
            "exfiltration": {
                "pattern": re.compile(
                    r"\b(exfiltrat(?:e|ion)|transfer|upload|steal|extract)",
                    re.IGNORECASE,
                ),
                "techniques": ["T1048", "T1048.003", "T1041"],
            },
            "command_and_control": {
                "pattern": re.compile(
                    r"\b(command\s*(?:and|&)\s*control|c2|c&c|beacon|callback)",
                    re.IGNORECASE,
                ),
                "techniques": ["T1071", "T1071.001", "T1071.004", "T1095"],
            },
            "impact": {
                "pattern": re.compile(
                    r"\b(ransom|encrypt|wipe|corrupt|deny|dos|ddos|destruct)",
                    re.IGNORECASE,
                ),
                "techniques": ["T1486", "T1489", "T1490"],
            },
        }

        return patterns

    '''def extract_techniques(
        self, 
        text: str, 
        min_confidence: float = 0.1, 
        max_results: int = 10
    ) -> List[Dict]:
        """
        Extract techniques using enhanced rule-based matching
        
        Args:
            text: Input text to analyze
            min_confidence: Minimum confidence threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of technique matches with confidence scores
        """
        # Get matches using the most efficient method
        if self.use_aho_corasick:
            matches = self._extract_with_aho_corasick(text)
        else:
            matches = self._extract_with_regex(text)
            
        # Apply contextual pattern matching to enhance results
        if self.use_contextual_boost:
            self._apply_contextual_patterns(text, matches)
            
        # Convert matches to results with confidence scoring
        results = self._score_matches(matches)
        
        # Filter by confidence threshold
        results = [r for r in results if r["confidence"] >= min_confidence]
        
        # Sort by confidence (descending) and limit results
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:max_results]'''

    # Add this to EnhancedRuleBasedExtractor in src/api/v2/routes/rule_based_V2.py

    def extract_techniques(
        self,
        text: str,
        min_confidence: float = 0.1,
        max_results: int = 10,
        job_id: str = None,
    ) -> List[Dict]:
        """
        Extract techniques using enhanced rule-based matching with metrics recording

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

        # Record extractor execution details if metrics enabled
        extractor_id = None
        if metrics_recorder:
            parameters = {
                "min_confidence": min_confidence,
                "max_results": max_results,
                "use_aho_corasick": self.use_aho_corasick,
                "use_contextual_boost": self.use_contextual_boost,
            }
            extractor_id = metrics_recorder.record_extractor_result(
                extractor_name="enhanced_rule_based",
                raw_input=text,
                raw_output={},  # Will update this later
                execution_time_ms=0,  # Will update this later
                parameters=parameters,
            )

        # Get matches using the most efficient method
        match_start = time.time()
        if self.use_aho_corasick:
            matches = self._extract_with_aho_corasick(text)
        else:
            matches = self._extract_with_regex(text)
        match_time = int((time.time() - match_start) * 1000)

        # Record pattern matching performance
        if metrics_recorder and extractor_id:
            metrics_recorder.record_performance_benchmark(
                extractor_name="enhanced_rule_based",
                operation_type="pattern_matching",
                execution_time_ms=match_time,
                input_size=len(text),
            )

        # Apply contextual pattern matching to enhance results
        context_start = time.time()
        if self.use_contextual_boost:
            self._apply_contextual_patterns(text, matches)
        context_time = int((time.time() - context_start) * 1000)

        # Record contextual pattern matching performance
        if metrics_recorder and extractor_id:
            metrics_recorder.record_performance_benchmark(
                extractor_name="enhanced_rule_based",
                operation_type="contextual_matching",
                execution_time_ms=context_time,
                input_size=len(text),
            )

        # Convert matches to results with confidence scoring
        scoring_start = time.time()
        results = self._score_matches(matches)
        scoring_time = int((time.time() - scoring_start) * 1000)

        # Record scoring performance
        if metrics_recorder and extractor_id:
            metrics_recorder.record_performance_benchmark(
                extractor_name="enhanced_rule_based",
                operation_type="confidence_scoring",
                execution_time_ms=scoring_time,
                input_size=len(matches),
            )

        # Record keyword matches if metrics enabled
        if metrics_recorder and extractor_id:
            for tech_id, match_data in matches.items():
                # Process keywords
                keyword_data = []
                for keyword in match_data.get("keywords", []):
                    # Get match position if possible
                    match_pos = text.lower().find(keyword.lower())
                    # Get surrounding context
                    if match_pos >= 0:
                        start_pos = max(0, match_pos - 50)
                        end_pos = min(len(text), match_pos + len(keyword) + 50)
                        context = text[start_pos:end_pos]
                    else:
                        context = ""

                    keyword_data.append(
                        {
                            "text": keyword,
                            "position": match_pos if match_pos >= 0 else None,
                            "context": context,
                        }
                    )

                if keyword_data:
                    metrics_recorder.record_keywords(
                        extractor_id=extractor_id,
                        technique_id=tech_id,
                        keywords=keyword_data,
                    )

        # Filter by confidence threshold
        results = [r for r in results if r["confidence"] >= min_confidence]

        # Sort by confidence (descending) and limit results
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

    def _extract_with_aho_corasick(self, text: str) -> Dict[str, Dict]:
        """
        Extract techniques using Aho-Corasick algorithm

        Args:
            text: Input text

        Returns:
            Dictionary of technique matches with match data
        """
        matches = {}

        # Use Aho-Corasick automaton to find all matches in a single pass
        automaton_matches = self.automaton.search(text)

        for tech_id, pattern_matches in automaton_matches.items():
            # Count unique matched keywords
            unique_keywords = set(match[0] for match in pattern_matches)
            total_matches = len(pattern_matches)

            matches[tech_id] = {
                "count": total_matches,
                "keywords": list(unique_keywords),
                "positions": [pos for _, pos in pattern_matches],
                "context_boost": 1.0,  # Default boost factor
            }

        return matches

    def _extract_with_regex(self, text: str) -> Dict[str, Dict]:
        """
        Extract techniques using regex patterns (fallback method)

        Args:
            text: Input text

        Returns:
            Dictionary of technique matches with match data
        """
        matches = {}

        # Use regex patterns for matching (less efficient but more familiar)
        for tech_id, pattern in self.technique_patterns.items():
            # Find all matches
            all_matches = pattern.findall(text)

            if all_matches:
                # Count unique matched keywords
                unique_keywords = set(match.lower() for match in all_matches)
                total_matches = len(all_matches)

                matches[tech_id] = {
                    "count": total_matches,
                    "keywords": list(unique_keywords),
                    "context_boost": 1.0,  # Default boost factor
                }

        return matches

    def _apply_contextual_patterns(self, text: str, matches: Dict[str, Dict]) -> None:
        """
        Apply contextual patterns to enhance matching

        Args:
            text: Input text
            matches: Dictionary of current matches to update
        """
        # Apply context patterns to discover additional techniques
        for context_type, context_data in self.context_patterns.items():
            pattern = context_data["pattern"]
            context_matches = pattern.findall(text)

            if context_matches:
                # Get techniques associated with this context
                for tech_id in context_data["techniques"]:
                    # If technique already matched, boost confidence
                    if tech_id in matches:
                        matches[tech_id]["context_boost"] = min(
                            matches[tech_id].get("context_boost", 1.0) + 0.2, 1.5
                        )
                        matches[tech_id]["keywords"].extend(context_matches)
                    else:
                        # Add as a new match with lower confidence
                        matches[tech_id] = {
                            "count": len(context_matches),
                            "keywords": list(set(context_matches)),
                            "context_boost": 0.8,  # Context-only matches get lower boost
                            "context_match": True,  # Flag as context match
                        }

    def _score_matches(self, matches: Dict[str, Dict]) -> List[Dict]:
        """
        Convert matches to results with advanced confidence scoring

        Args:
            matches: Dictionary of technique matches

        Returns:
            List of results with confidence scores
        """
        results = []

        for tech_id, match_data in matches.items():
            # Get match data
            match_count = match_data["count"]
            keywords = match_data.get("keywords", [])
            unique_keywords = len(set(keywords))
            context_boost = match_data.get("context_boost", 1.0)
            is_context_match = match_data.get("context_match", False)

            # Calculate base confidence based on number of unique keywords and matches
            # More unique keywords = higher confidence
            if is_context_match:
                # Context-only matches start with lower base confidence
                base_confidence = 0.3 + (unique_keywords * 0.05)
            else:
                # Regular keyword matches get higher base confidence
                base_confidence = 0.4 + (unique_keywords * 0.08)

                # Boost for multiple matches of the same keyword
                if match_count > unique_keywords:
                    repetition_factor = min(match_count / max(unique_keywords, 1), 3)
                    base_confidence += 0.05 * repetition_factor

            # Apply context boost
            final_confidence = min(base_confidence * context_boost, 0.95)

            # Add keyword density factor - if keywords are clustered, likely more relevant
            if "positions" in match_data and len(match_data["positions"]) > 1:
                positions = sorted(match_data["positions"])
                avg_distance = np.mean(
                    [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
                )

                # If keywords are close together, boost confidence
                if avg_distance < 100:  # Within 100 characters
                    density_boost = 1.0 + (0.1 * (1 - (avg_distance / 100)))
                    final_confidence = min(final_confidence * density_boost, 0.95)

            # Create result
            result = {
                "technique_id": tech_id,
                "confidence": final_confidence,
                "match_count": match_count,
                "matched_keywords": list(set(keywords)),
                "method": "enhanced_rule_based",
            }

            # Add technique name if available
            if tech_id in self.techniques_data:
                result["name"] = self.techniques_data[tech_id].get("name", "")

            results.append(result)

        return results
