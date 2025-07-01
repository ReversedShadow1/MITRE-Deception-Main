"""
Enhanced BM25 Extractor for ATT&CK Techniques
-------------------------------------------
Implements vectorized BM25 ranking with security-focused tokenization and field weighting.
"""

import json
import logging
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from src.database.postgresql import get_db

logger = logging.getLogger("EnhancedBM25Extractor")


class SecurityTokenizer:
    """
    Custom tokenizer for security text that preserves technical terms
    and handles security-specific vocabulary
    """

    def __init__(self):
        """Initialize the security tokenizer with specialized patterns"""
        # Security-specific patterns to preserve
        self.preserve_patterns = [
            # CVE IDs
            r"CVE-\d{4}-\d{1,7}",
            # CWE IDs
            r"CWE-\d+",
            # ATT&CK Technique IDs
            r"T\d{4}(?:\.\d{3})?",
            # Common security abbreviations (preserve as single tokens)
            r"\b(XSS|CSRF|RCE|SQLi|SSRF|XXE|IDOR|MITM|DoS|DDoS|APT|C2|TTP)\b",
        ]

        # Combined pattern for preservation
        self.preserve_regex = re.compile(
            "|".join(self.preserve_patterns), re.IGNORECASE
        )

        # Common security stopwords to remove
        self.security_stopwords = {
            "the",
            "and",
            "or",
            "a",
            "an",
            "is",
            "it",
            "this",
            "that",
            "these",
            "those",
            "be",
            "to",
            "of",
            "in",
            "for",
            "with",
            "by",
            "at",
            "on",
            "from",
        }

        # Security term synonyms for expansion
        self.security_synonyms = {
            "attacker": ["adversary", "threat actor", "hacker", "malicious actor"],
            "malware": ["malicious software", "malicious code", "virus", "trojan"],
            "exfiltrate": ["steal", "extract", "leak", "transfer", "upload"],
            "vulnerability": ["weakness", "security flaw", "exploit", "bug"],
            "c2": ["command and control", "c&c", "command & control"],
            "lateral movement": ["pivoting", "internal movement", "east-west movement"],
            "persistence": [
                "maintain access",
                "maintain presence",
                "autostart",
                "startup",
            ],
            "privilege escalation": [
                "privesc",
                "elevation of privilege",
                "root access",
            ],
        }

        # Build reverse lookup for synonyms
        self.reverse_synonyms = {}
        for term, synonyms in self.security_synonyms.items():
            for synonym in synonyms:
                self.reverse_synonyms[synonym.lower()] = term.lower()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with security focus

        Args:
            text: Input text

        Returns:
            List of security-focused tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Find all preserved patterns and temporarily replace them
        preserved = {}

        def replace_preserved(match):
            placeholder = f"__PRESERVED_{len(preserved)}__"
            preserved[placeholder] = match.group(0).lower()
            return placeholder

        text = self.preserve_regex.sub(replace_preserved, text)

        # Tokenize remaining text (simple whitespace/punctuation splitting for now)
        tokens = []
        for word in re.findall(r"\b\w+\b", text):
            # Skip stopwords
            if word in self.security_stopwords:
                continue

            # Normalize synonyms
            if word in self.reverse_synonyms:
                tokens.append(self.reverse_synonyms[word])
            else:
                tokens.append(word)

        # Restore preserved patterns
        for i, token in enumerate(tokens):
            if token in preserved:
                tokens[i] = preserved[token]

        # Add preserved patterns that might have been missed
        for placeholder, value in preserved.items():
            if placeholder in text and value not in tokens:
                tokens.append(value)

        return tokens

    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize a batch of texts

        Args:
            texts: List of input texts

        Returns:
            List of token lists
        """
        return [self.tokenize(text) for text in texts]


class EnhancedBM25Extractor:
    """
    Enhanced BM25 extractor for ATT&CK techniques using vectorized operations
    and security-focused tokenization
    """

    def __init__(
        self,
        techniques: Dict,
        technique_keywords: Dict,
        models_dir: str = "models/bm25_enhanced",
        bm25_variant: str = "plus",
        use_cache: bool = True,
        use_field_weighting: bool = True,
        neo4j_connector=None,
    ):
        """
        Initialize enhanced BM25 extractor

        Args:
            techniques: Dictionary of technique data
            technique_keywords: Dictionary of technique keywords
            models_dir: Directory for model storage
            bm25_variant: BM25 variant to use ('okapi' or 'plus')
            use_cache: Whether to use cached model
            use_field_weighting: Whether to use field weighting
            neo4j_connector: Optional Neo4j connector
        """
        self.techniques = techniques
        self.technique_keywords = technique_keywords
        self.models_dir = models_dir
        self.bm25_variant = bm25_variant.lower()
        self.use_cache = use_cache
        self.use_field_weighting = use_field_weighting
        self.neo4j_connector = neo4j_connector

        # Create model directory
        os.makedirs(models_dir, exist_ok=True)

        # Initialize security tokenizer
        self.tokenizer = SecurityTokenizer()

        # Set cache file paths
        self.corpus_cache_path = os.path.join(models_dir, "corpus.json")
        self.model_cache_path = os.path.join(
            models_dir, f"bm25_{bm25_variant}_model.pkl"
        )
        self.vectorizer_cache_path = os.path.join(models_dir, "vectorizer.pkl")

        # Initialize model components
        self.corpus = []
        self.corpus_fields = []  # Track field type for each document
        self.tokenized_corpus = []
        self.tech_ids = []
        self.bm25_model = None
        self.vectorizer = None

        # Field weights for different parts of technique data
        self.field_weights = {
            "name": 3.0,  # Technique name (highest weight)
            "description": 1.5,  # Description (medium weight)
            "keyword": 2.0,  # Keywords (high weight)
        }

        # Auto-initialize
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the BM25 model"""
        # Try loading from cache first
        if self.use_cache and self._load_cached_model():
            return

        # Build from scratch otherwise
        self._build_model()

    def _load_cached_model(self) -> bool:
        """
        Load model from cache

        Returns:
            Whether load was successful
        """
        cache_files = [self.corpus_cache_path, self.model_cache_path]

        if self.use_field_weighting:
            cache_files.append(self.vectorizer_cache_path)

        if not all(os.path.exists(f) for f in cache_files):
            return False

        try:
            # Load corpus and technique IDs
            with open(self.corpus_cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                self.corpus = cache_data.get("corpus", [])
                self.tech_ids = cache_data.get("tech_ids", [])
                self.corpus_fields = cache_data.get("corpus_fields", [])

                # Tokenize corpus if not using vectorizer
                if not self.use_field_weighting:
                    self.tokenized_corpus = self.tokenizer.batch_tokenize(self.corpus)

            # Load BM25 model
            with open(self.model_cache_path, "rb") as f:
                self.bm25_model = pickle.load(f)

            # Load vectorizer if using field weighting
            if self.use_field_weighting and os.path.exists(self.vectorizer_cache_path):
                with open(self.vectorizer_cache_path, "rb") as f:
                    self.vectorizer = pickle.load(f)

            logger.info(
                f"Loaded enhanced BM25 model from cache with {len(self.corpus)} documents"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load cached BM25 model: {str(e)}")
            return False

    def _build_model(self) -> None:
        """Build enhanced BM25 model from technique data"""
        logger.info("Building enhanced BM25 model from scratch...")

        self.corpus = []
        self.corpus_fields = []
        self.tech_ids = []

        # Build corpus from technique data and keywords with field tracking
        for tech_id, tech_data in self.techniques.items():
            # Skip techniques without keywords
            if tech_id not in self.technique_keywords:
                continue

            # Get technique data
            tech_name = tech_data.get("name", "")
            tech_desc = tech_data.get("description", "")
            tech_keywords = self.technique_keywords.get(tech_id, [])

            # Add name as separate document with field tracking
            if tech_name:
                self.corpus.append(tech_name)
                self.tech_ids.append(tech_id)
                self.corpus_fields.append("name")

            # Add description (first 2 sentences) with field tracking
            if tech_desc:
                # Extract first two sentences or 200 chars
                first_sentences = ".".join(tech_desc.split(".")[:2])
                if first_sentences:
                    self.corpus.append(first_sentences)
                    self.tech_ids.append(tech_id)
                    self.corpus_fields.append("description")

            # Add each keyword as separate document with field tracking
            for keyword in tech_keywords:
                self.corpus.append(keyword)
                self.tech_ids.append(tech_id)
                self.corpus_fields.append("keyword")

        # Process differently depending on whether field weighting is used
        if self.use_field_weighting:
            self._build_vectorized_model()
        else:
            self._build_standard_model()

        logger.info(f"Built enhanced BM25 model with {len(self.corpus)} documents")

        # Cache model if enabled
        if self.use_cache:
            self._cache_model()

    def _build_vectorized_model(self) -> None:
        """Build vectorized BM25 model with field weighting"""
        # Create and fit the vectorizer
        self.vectorizer = CountVectorizer(
            tokenizer=self.tokenizer.tokenize,
            lowercase=False,  # Already handled in tokenizer
            min_df=1,  # Keep all terms
            binary=False,  # Count occurrences
        )

        # Convert corpus to document-term matrix
        X = self.vectorizer.fit_transform(self.corpus)

        # Create BM25 model
        if self.bm25_variant == "plus":
            self.bm25_model = VectorizedBM25Plus(
                X, self.corpus_fields, self.field_weights
            )
        else:
            self.bm25_model = VectorizedBM25Okapi(
                X, self.corpus_fields, self.field_weights
            )

    def _build_standard_model(self) -> None:
        """Build standard BM25 model without field weighting"""
        # Tokenize corpus
        self.tokenized_corpus = self.tokenizer.batch_tokenize(self.corpus)

        # Create BM25 model
        if self.bm25_variant == "plus":
            self.bm25_model = BM25Plus(self.tokenized_corpus)
        else:
            self.bm25_model = BM25Okapi(self.tokenized_corpus)

    def _cache_model(self) -> None:
        """Cache BM25 model to disk"""
        try:
            # Save corpus and tech IDs
            with open(self.corpus_cache_path, "w", encoding="utf-8") as f:
                cache_data = {
                    "corpus": self.corpus,
                    "tech_ids": self.tech_ids,
                    "corpus_fields": self.corpus_fields,
                }
                json.dump(cache_data, f)

            # Save BM25 model
            with open(self.model_cache_path, "wb") as f:
                pickle.dump(self.bm25_model, f)

            # Save vectorizer if using field weighting
            if self.use_field_weighting and self.vectorizer:
                with open(self.vectorizer_cache_path, "wb") as f:
                    pickle.dump(self.vectorizer, f)

            logger.info("Cached enhanced BM25 model to disk")

        except Exception as e:
            logger.error(f"Failed to cache BM25 model: {str(e)}")


    def extract_techniques(
        self, text: str, threshold: float = 0.1, top_k: int = 10, job_id: str = None
    ) -> List[Dict]:
        """
        Extract techniques using enhanced BM25 ranking with metrics recording

        Args:
            text: Input text to analyze
            threshold: Minimum score threshold (0-1)
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

        if not self.bm25_model:
            logger.error("BM25 model not initialized")
            return []

        # Record extractor execution details if metrics enabled
        extractor_id = None
        if metrics_recorder:
            parameters = {
                "threshold": threshold,
                "top_k": top_k,
                "bm25_variant": self.bm25_variant,
                "use_field_weighting": self.use_field_weighting,
            }
            extractor_id = metrics_recorder.record_extractor_result(
                extractor_name="enhanced_bm25",
                raw_input=text,
                raw_output={},  # Will update this later
                execution_time_ms=0,  # Will update this later
                parameters=parameters,
            )

            # Record text segment
            metrics_recorder.record_text_segment(text=text, index=0)

        # Record tokenization time
        tokenize_start = time.time()

        # Process differently depending on whether vectorizer is used
        if self.use_field_weighting and self.vectorizer:
            # Tokenize and vectorize text
            query_vector = self.vectorizer.transform([text])
            tokenize_time = int((time.time() - tokenize_start) * 1000)

            # Record tokenization performance
            if metrics_recorder and extractor_id:
                metrics_recorder.record_performance_benchmark(
                    extractor_name="enhanced_bm25",
                    operation_type="text_vectorization",
                    execution_time_ms=tokenize_time,
                    input_size=len(text),
                )

            # Get BM25 scores (vectorized operation)
            score_start = time.time()
            scores = self.bm25_model.get_scores(query_vector)
            score_time = int((time.time() - score_start) * 1000)

            # Record scoring performance
            if metrics_recorder and extractor_id:
                metrics_recorder.record_performance_benchmark(
                    extractor_name="enhanced_bm25",
                    operation_type="bm25_scoring",
                    execution_time_ms=score_time,
                    input_size=len(text),
                )

            # Process scores to get results
            results = self._process_scores(scores, text, threshold, top_k)
        else:
            # Tokenize query
            query_tokens = self.tokenizer.tokenize(text)
            tokenize_time = int((time.time() - tokenize_start) * 1000)

            # Record tokenization performance
            if metrics_recorder and extractor_id:
                metrics_recorder.record_performance_benchmark(
                    extractor_name="enhanced_bm25",
                    operation_type="text_tokenization",
                    execution_time_ms=tokenize_time,
                    input_size=len(text),
                )

                # Record tokenized terms
                token_entities = []
                for token in query_tokens:
                    token_entities.append(
                        {
                            "text": token,
                            "type": "TOKEN",
                            "start_offset": text.lower().find(token.lower()),
                            "end_offset": None,
                        }
                    )

                if token_entities:
                    metrics_recorder.record_entities(
                        extractor_id=extractor_id, entities=token_entities
                    )

            # Get BM25 scores
            score_start = time.time()
            scores = self.bm25_model.get_scores(query_tokens)
            score_time = int((time.time() - score_start) * 1000)

            # Record scoring performance
            if metrics_recorder and extractor_id:
                metrics_recorder.record_performance_benchmark(
                    extractor_name="enhanced_bm25",
                    operation_type="bm25_scoring",
                    execution_time_ms=score_time,
                    input_size=len(query_tokens),
                )

            # Process scores to get results
            results = self._process_scores(scores, text, threshold, top_k)

        # Record BM25 scores if metrics enabled
        if metrics_recorder and extractor_id:
            for result in results:
                tech_id = result.get("technique_id")
                confidence = result.get("confidence", 0.0)
                raw_score = result.get("raw_score", 0.0)

                # Find which terms matched this technique
                matched_terms = {}
                if "matched_keywords" in result:
                    for keyword in result["matched_keywords"]:
                        matched_terms[keyword] = text.lower().count(keyword.lower())

                metrics_recorder.record_bm25_scores(
                    extractor_id=extractor_id,
                    technique_id=tech_id,
                    raw_score=raw_score,
                    normalized_score=confidence,
                    matched_terms=matched_terms,
                )

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

    def _extract_vectorized(
        self, text: str, threshold: float = 0.1, top_k: int = 10
    ) -> List[Dict]:
        """
        Extract techniques using vectorized BM25

        Args:
            text: Input text to analyze
            threshold: Minimum score threshold (0-1)
            top_k: Maximum number of results

        Returns:
            List of technique matches with scores
        """
        # Convert query to document-term vector
        query_vector = self.vectorizer.transform([text])

        # Get BM25 scores (vectorized operation)
        scores = self.bm25_model.get_scores(query_vector)

        # Process scores to get results
        return self._process_scores(scores, text, threshold, top_k)

    def _extract_standard(
        self, text: str, threshold: float = 0.1, top_k: int = 10
    ) -> List[Dict]:
        """
        Extract techniques using standard BM25

        Args:
            text: Input text to analyze
            threshold: Minimum score threshold (0-1)
            top_k: Maximum number of results

        Returns:
            List of technique matches with scores
        """
        # Tokenize query
        query_tokens = self.tokenizer.tokenize(text)

        # Get BM25 scores
        scores = self.bm25_model.get_scores(query_tokens)

        # Process scores to get results
        return self._process_scores(scores, text, threshold, top_k)

    def _process_scores(
        self, scores: np.ndarray, text: str, threshold: float, top_k: int
    ) -> List[Dict]:
        """
        Process BM25 scores to get technique results

        Args:
            scores: Array of BM25 scores
            text: Original query text (for debugging)
            threshold: Minimum score threshold
            top_k: Maximum number of results

        Returns:
            List of technique matches with scores
        """
        # Find max score for normalization
        max_score = max(scores) if len(scores) > 0 else 1.0

        # Group scores by technique ID
        tech_scores = {}

        for i, score in enumerate(scores):
            if i >= len(self.tech_ids):
                continue

            tech_id = self.tech_ids[i]

            # Skip zero scores
            if score == 0:
                continue

            # Get field type for weighting if available
            field_type = (
                self.corpus_fields[i] if i < len(self.corpus_fields) else "unknown"
            )

            # Store score if higher than existing
            if tech_id not in tech_scores or score > tech_scores[tech_id]["score"]:
                tech_scores[tech_id] = {"score": score, "field": field_type}

        # Convert to results
        results = []

        for tech_id, score_data in tech_scores.items():
            score = score_data["score"]
            field = score_data["field"]

            # Normalize score
            normalized_score = score / max_score if max_score > 0 else 0

            if normalized_score >= threshold:
                # Create result
                result = {
                    "technique_id": tech_id,
                    "confidence": float(normalized_score),
                    "raw_score": float(score),
                    "method": "enhanced_bm25",
                    "matched_field": field,
                }

                # Add matched keywords when available
                if tech_id in self.technique_keywords:
                    # Find which keywords match the text
                    matched_keywords = []
                    for keyword in self.technique_keywords[tech_id]:
                        if keyword.lower() in text.lower():
                            matched_keywords.append(keyword)

                    if matched_keywords:
                        result["matched_keywords"] = matched_keywords

                # Add technique name if available
                if tech_id in self.techniques:
                    result["name"] = self.techniques[tech_id].get("name", "")

                results.append(result)

        # Sort by confidence and limit to top_k
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

    def clear_cache(self) -> None:
        """Clear the model cache"""
        cache_files = [
            self.corpus_cache_path,
            self.model_cache_path,
            self.vectorizer_cache_path,
        ]

        for file_path in cache_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Removed cache file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove cache file {file_path}: {str(e)}")


class VectorizedBM25Base:
    """Base class for vectorized BM25 implementations"""

    def __init__(self, X, corpus_fields=None, field_weights=None):
        """
        Initialize vectorized BM25

        Args:
            X: Document-term matrix (sparse)
            corpus_fields: List of field types for each document
            field_weights: Dictionary of field weights
        """
        self.X = X
        self.corpus_fields = corpus_fields or []
        self.field_weights = field_weights or {}

        # Calculate document lengths and average document length
        self.doc_lens = np.asarray(X.sum(axis=1)).flatten()
        self.avgdl = np.mean(self.doc_lens)

        # Get vocabulary size
        self.vocab_size = X.shape[1]

        # Calculate document frequency for each term
        self.df = np.diff(X.tocsc().indptr)

        # Calculate IDF values
        self.idf = self._calculate_idf()

        # Prepare field weight vector if fields provided
        if corpus_fields and field_weights:
            self.field_weight_vector = np.ones(len(corpus_fields))
            for i, field in enumerate(corpus_fields):
                if field in field_weights:
                    self.field_weight_vector[i] = field_weights[field]
        else:
            self.field_weight_vector = None

    def _calculate_idf(self) -> np.ndarray:
        """
        Calculate IDF values for terms

        Returns:
            Array of IDF values
        """
        # To be implemented by subclasses
        raise NotImplementedError()

    def get_scores(self, query_vector) -> np.ndarray:
        """
        Get BM25 scores for query

        Args:
            query_vector: Query vector (sparse)

        Returns:
            Array of scores
        """
        # To be implemented by subclasses
        raise NotImplementedError()


class VectorizedBM25Okapi(VectorizedBM25Base):
    """Vectorized implementation of BM25 Okapi"""

    def __init__(self, X, corpus_fields=None, field_weights=None, k1=1.5, b=0.75):
        """
        Initialize vectorized BM25 Okapi

        Args:
            X: Document-term matrix (sparse)
            corpus_fields: List of field types for each document
            field_weights: Dictionary of field weights
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        super().__init__(X, corpus_fields, field_weights)

    def _calculate_idf(self) -> np.ndarray:
        """
        Calculate Okapi BM25 IDF values

        Returns:
            Array of IDF values
        """
        # Number of documents
        n_docs = self.X.shape[0]

        # Calculate IDF using Okapi BM25 formula
        # log((N - n + 0.5) / (n + 0.5))
        return np.log((n_docs - self.df + 0.5) / (self.df + 0.5) + 1e-10)

    def get_scores(self, query_vector) -> np.ndarray:
        """
        Get BM25 scores for query

        Args:
            query_vector: Query vector (sparse)

        Returns:
            Array of scores
        """
        # Extract query term frequencies
        query_terms = query_vector.indices
        query_tf = query_vector.data

        # Initialize scores array
        scores = np.zeros(self.X.shape[0])

        # Get document frequencies for query terms
        doc_vectors = self.X[:, query_terms]

        # Convert to CSR for efficient row slicing
        if not sparse.isspmatrix_csr(doc_vectors):
            doc_vectors = doc_vectors.tocsr()

        # Calculate scores using vectorized operations
        for i, (idx, tf) in enumerate(zip(query_terms, query_tf)):
            # Skip terms not in vocabulary
            if idx >= self.vocab_size:
                continue

            # Get term frequency in documents
            term_docs = doc_vectors.getcol(i).toarray().flatten()

            # BM25 score for this term
            idf = self.idf[idx]
            doc_len_norm = 1 - self.b + self.b * (self.doc_lens / self.avgdl)
            term_scores = (
                idf * (term_docs * (self.k1 + 1)) / (term_docs + self.k1 * doc_len_norm)
            )

            # Add to total scores
            scores += term_scores

        # Apply field weights if available
        if self.field_weight_vector is not None:
            scores *= self.field_weight_vector

        return scores


class VectorizedBM25Plus(VectorizedBM25Base):
    """Vectorized implementation of BM25Plus"""

    def __init__(
        self, X, corpus_fields=None, field_weights=None, k1=1.5, b=0.75, delta=1.0
    ):
        """
        Initialize vectorized BM25Plus

        Args:
            X: Document-term matrix (sparse)
            corpus_fields: List of field types for each document
            field_weights: Dictionary of field weights
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            delta: Additional parameter for BM25+
        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(X, corpus_fields, field_weights)

    def _calculate_idf(self) -> np.ndarray:
        """
        Calculate BM25+ IDF values

        Returns:
            Array of IDF values
        """
        # Number of documents
        n_docs = self.X.shape[0]

        # Calculate IDF using standard formula: log(N/n)
        return np.log(1 + (n_docs - self.df + 0.5) / (self.df + 0.5))

    def get_scores(self, query_vector) -> np.ndarray:
        """
        Get BM25+ scores for query

        Args:
            query_vector: Query vector (sparse)

        Returns:
            Array of scores
        """
        # Extract query term frequencies
        query_terms = query_vector.indices
        query_tf = query_vector.data

        # Initialize scores array
        scores = np.zeros(self.X.shape[0])

        # Get document frequencies for query terms
        doc_vectors = self.X[:, query_terms]

        # Convert to CSR for efficient row slicing
        if not sparse.isspmatrix_csr(doc_vectors):
            doc_vectors = doc_vectors.tocsr()

        # Calculate scores using vectorized operations
        for i, (idx, tf) in enumerate(zip(query_terms, query_tf)):
            # Skip terms not in vocabulary
            if idx >= self.vocab_size:
                continue

            # Get term frequency in documents
            term_docs = doc_vectors.getcol(i).toarray().flatten()

            # BM25+ score for this term
            idf = self.idf[idx]
            doc_len_norm = 1 - self.b + self.b * (self.doc_lens / self.avgdl)
            term_scores = (
                idf * (term_docs * (self.k1 + 1)) / (term_docs + self.k1 * doc_len_norm)
                + self.delta
            )

            # Add to total scores
            scores += term_scores

        # Apply field weights if available
        if self.field_weight_vector is not None:
            scores *= self.field_weight_vector

        return scores
