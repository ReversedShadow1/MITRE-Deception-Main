import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple


class LargeTextProcessor:
    """
    Processor for chunking large text inputs and processing them efficiently
    """

    def __init__(
        self,
        chunk_size: int = 5000,  # Characters per chunk
        chunk_overlap: int = 500,  # Overlap between chunks
        max_single_chunk_size: int = 25000,  # Maximum size for single-chunk processing
        boundary_markers: Set[str] = None,  # Preferred chunk boundary markers
    ):
        """
        Initialize text processor

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Overlap size between chunks to maintain context
            max_single_chunk_size: Maximum size to process without chunking
            boundary_markers: Set of preferred boundary markers for clean breaks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_single_chunk_size = max_single_chunk_size
        self.boundary_markers = boundary_markers or {
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ":",
            " - ",
            "--",
        }

    def should_chunk(self, text: str) -> bool:
        """
        Determine if text needs chunking

        Args:
            text: Input text

        Returns:
            Whether text should be chunked
        """
        return len(text) > self.max_single_chunk_size

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks at natural boundaries

        Args:
            text: Input text

        Returns:
            List of chunks with metadata
        """
        if not self.should_chunk(text):
            # Return as single chunk if under threshold
            return [
                {
                    "chunk_id": str(uuid.uuid4()),
                    "text": text,
                    "start_pos": 0,
                    "end_pos": len(text),
                    "is_first_chunk": True,
                    "is_last_chunk": True,
                }
            ]

        chunks = []
        position = 0
        text_length = len(text)

        while position < text_length:
            # Calculate end position for current chunk
            chunk_end = min(position + self.chunk_size, text_length)

            # Find a natural boundary to end the chunk if not the last chunk
            if chunk_end < text_length:
                # Look for boundary markers within the last portion of the chunk
                search_start = chunk_end - min(500, self.chunk_size // 4)
                best_boundary = chunk_end

                # Try each boundary marker in order of preference
                for marker in self.boundary_markers:
                    # Find last occurrence of marker in search region
                    boundary_pos = text.rfind(marker, search_start, chunk_end)
                    if boundary_pos > search_start:
                        # Found a good boundary
                        best_boundary = boundary_pos + len(marker)
                        break

                chunk_end = best_boundary

            # Extract chunk
            chunk_text = text[position:chunk_end]

            # Create chunk with metadata
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "start_pos": position,
                    "end_pos": chunk_end,
                    "is_first_chunk": position == 0,
                    "is_last_chunk": chunk_end >= text_length,
                }
            )

            # Move position for next chunk, ensuring overlap
            position = chunk_end - self.chunk_overlap

            # Ensure we make progress
            if position <= chunks[-1]["start_pos"]:
                position = chunks[-1]["end_pos"]

        return chunks

    def process_chunks_with_extractor(
        self, chunks: List[Dict[str, Any]], extractor_func, **extractor_kwargs
    ) -> Tuple[List[Dict], Dict]:
        """
        Process chunks with an extractor function and merge results

        Args:
            chunks: List of text chunks
            extractor_func: Function to extract entities from each chunk
            **extractor_kwargs: Additional arguments for extractor function

        Returns:
            Tuple of (merged results, metadata)
        """
        # Extract from each chunk
        chunk_results = []
        processing_stats = {
            "total_chunks": len(chunks),
            "chunk_sizes": [],
            "processing_times": [],
            "techniques_per_chunk": [],
        }

        for chunk in chunks:
            # Process chunk
            import time

            start_time = time.time()

            # Add chunk metadata to kwargs
            kwargs = {**extractor_kwargs}
            kwargs["chunk_metadata"] = {
                "chunk_id": chunk["chunk_id"],
                "is_first_chunk": chunk["is_first_chunk"],
                "is_last_chunk": chunk["is_last_chunk"],
                "original_position": (chunk["start_pos"], chunk["end_pos"]),
            }

            # Call extractor function
            result = extractor_func(chunk["text"], **kwargs)

            # Collect statistics
            processing_time = time.time() - start_time
            processing_stats["chunk_sizes"].append(len(chunk["text"]))
            processing_stats["processing_times"].append(processing_time)

            # Add position information to results
            if "techniques" in result:
                techniques_count = len(result["techniques"])
                processing_stats["techniques_per_chunk"].append(techniques_count)

                # Add position offset to matched keywords
                for technique in result["techniques"]:
                    if "matched_keywords" in technique and isinstance(
                        technique["matched_keywords"], list
                    ):
                        # If matched_keywords contains position info
                        for i, keyword in enumerate(technique["matched_keywords"]):
                            if isinstance(keyword, dict) and "position" in keyword:
                                # Adjust position
                                keyword["position"] += chunk["start_pos"]
                                keyword["chunk_id"] = chunk["chunk_id"]

            # Add chunk result
            chunk_results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "result": result,
                    "processing_time": processing_time,
                }
            )

        # Merge results
        merged_results = self._merge_chunk_results(chunk_results)

        # Calculate aggregate statistics
        if processing_stats["processing_times"]:
            processing_stats["avg_chunk_size"] = sum(
                processing_stats["chunk_sizes"]
            ) / len(processing_stats["chunk_sizes"])
            processing_stats["avg_processing_time"] = sum(
                processing_stats["processing_times"]
            ) / len(processing_stats["processing_times"])
            processing_stats["total_processing_time"] = sum(
                processing_stats["processing_times"]
            )

            if processing_stats["techniques_per_chunk"]:
                processing_stats["avg_techniques_per_chunk"] = sum(
                    processing_stats["techniques_per_chunk"]
                ) / len(processing_stats["techniques_per_chunk"])
                processing_stats["total_techniques"] = len(
                    merged_results.get("techniques", [])
                )

        return merged_results, processing_stats

    def _merge_chunk_results(self, chunk_results: List[Dict]) -> Dict:
        """
        Merge results from multiple chunks

        Args:
            chunk_results: List of results from individual chunks

        Returns:
            Merged results
        """
        if not chunk_results:
            return {"techniques": [], "meta": {"chunked": False}}

        # Initialize with first chunk's result structure
        merged = {
            "techniques": [],
            "meta": {
                "chunked": True,
                "chunk_count": len(chunk_results),
                "processing_time": sum(cr["processing_time"] for cr in chunk_results),
            },
        }

        # Collect all techniques
        techniques_by_id = {}

        for chunk_result in chunk_results:
            result = chunk_result["result"]

            # Copy meta information from first chunk
            if "meta" in result and len(merged["meta"]) <= 2:
                for key, value in result["meta"].items():
                    if key not in merged["meta"]:
                        merged["meta"][key] = value

            # Process techniques
            if "techniques" in result:
                for technique in result["techniques"]:
                    technique_id = technique.get("technique_id")
                    if not technique_id:
                        continue

                    # Check if we've seen this technique before
                    if technique_id in techniques_by_id:
                        existing = techniques_by_id[technique_id]

                        # Merge confidence - take max confidence
                        existing["confidence"] = max(
                            existing.get("confidence", 0),
                            technique.get("confidence", 0),
                        )

                        # Merge matched keywords
                        if "matched_keywords" in technique:
                            if "matched_keywords" not in existing:
                                existing["matched_keywords"] = []

                            # Add new keywords without duplicates
                            existing_keywords = set(
                                k["text"] if isinstance(k, dict) else k
                                for k in existing["matched_keywords"]
                            )

                            for keyword in technique["matched_keywords"]:
                                key_text = (
                                    keyword["text"]
                                    if isinstance(keyword, dict)
                                    else keyword
                                )
                                if key_text not in existing_keywords:
                                    existing["matched_keywords"].append(keyword)
                                    existing_keywords.add(key_text)

                        # Merge matched entities
                        if "matched_entities" in technique:
                            if "matched_entities" not in existing:
                                existing["matched_entities"] = []

                            # Add new entities without duplicates
                            existing_entities = set(
                                e["text"] if isinstance(e, dict) else e
                                for e in existing["matched_entities"]
                            )

                            for entity in technique["matched_entities"]:
                                entity_text = (
                                    entity["text"]
                                    if isinstance(entity, dict)
                                    else entity
                                )
                                if entity_text not in existing_entities:
                                    existing["matched_entities"].append(entity)
                                    existing_entities.add(entity_text)

                        # Merge component scores if available
                        if "component_scores" in technique:
                            if "component_scores" not in existing:
                                existing["component_scores"] = {}

                            for component, score in technique[
                                "component_scores"
                            ].items():
                                if component not in existing["component_scores"]:
                                    existing["component_scores"][component] = score
                                else:
                                    existing["component_scores"][component] = max(
                                        existing["component_scores"][component], score
                                    )
                    else:
                        # First time seeing this technique
                        techniques_by_id[technique_id] = technique.copy()

                        # Add chunk info
                        if "chunk_id" not in technique:
                            technique["chunk_id"] = chunk_result["chunk_id"]

        # Add all techniques to merged result, sorted by confidence
        merged["techniques"] = sorted(
            techniques_by_id.values(),
            key=lambda x: x.get("confidence", 0),
            reverse=True,
        )

        return merged
