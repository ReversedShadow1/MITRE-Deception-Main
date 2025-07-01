"""
Result Caching Layer for ATT&CK Extractor
---------------------------------------
Implements efficient caching of extraction results to improve performance.
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import redis

logger = logging.getLogger("ResultCache")


class ExtractionResultCache:
    """
    Cache for ATT&CK technique extraction results

    Supports multiple backends:
    - In-memory (default)
    - File-based
    - Redis (if available)
    """

    def __init__(
        self,
        cache_type: str = "memory",
        cache_dir: str = "cache/extraction",
        redis_url: str = None,
        ttl: int = 86400,  # 24 hours default TTL
        cache_size_limit: int = 1000,  # Maximum entries for memory cache
        text_hash_method: str = "md5",
    ):
        """
        Initialize the result cache

        Args:
            cache_type: Type of cache ('memory', 'file', or 'redis')
            cache_dir: Directory for file cache
            redis_url: URL for Redis connection
            ttl: Time-to-live for cache entries in seconds
            cache_size_limit: Maximum entries for memory cache
            text_hash_method: Method for hashing text keys ('md5' or 'sha256')
        """
        self.cache_type = cache_type.lower()
        self.cache_dir = cache_dir
        self.redis_url = redis_url
        self.ttl = ttl
        self.cache_size_limit = cache_size_limit
        self.text_hash_method = text_hash_method

        # Initialize appropriate cache backend
        if self.cache_type == "memory":
            self.cache = {}
            self.timestamps = {}
            self.access_counts = {}
        elif self.cache_type == "file":
            os.makedirs(cache_dir, exist_ok=True)
        elif self.cache_type == "redis":
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_prefix = "attack_extractor:"
                logger.info(f"Connected to Redis cache at {redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                logger.info("Falling back to in-memory cache")
                self.cache_type = "memory"
                self.cache = {}
                self.timestamps = {}
                self.access_counts = {}
        else:
            logger.warning(f"Unknown cache type: {cache_type}, using memory cache")
            self.cache_type = "memory"
            self.cache = {}
            self.timestamps = {}
            self.access_counts = {}

        logger.info(f"Initialized {self.cache_type} cache with {ttl}s TTL")

    def _hash_text(self, text: str) -> str:
        """
        Create a hash of the input text for use as cache key

        Args:
            text: Input text

        Returns:
            Hash string
        """
        # Add normalization to ensure consistent hashing
        text = text.strip().lower()

        if self.text_hash_method == "sha256":
            return hashlib.sha256(text.encode()).hexdigest()
        else:
            return hashlib.md5(text.encode()).hexdigest()

    def _build_cache_key(
        self,
        text: str,
        extractors: List[str] = None,
        threshold: float = 0.2,
        top_k: int = 10,
        use_ensemble: bool = True,
    ) -> str:
        """
        Build a cache key from extraction parameters

        Args:
            text: Input text
            extractors: List of extractors
            threshold: Confidence threshold
            top_k: Maximum results
            use_ensemble: Whether ensemble was used

        Returns:
            Cache key string
        """
        # Create consistent representation of parameters
        text_hash = self._hash_text(text)
        extractors_str = ",".join(sorted(extractors)) if extractors else "default"
        params_str = f"{threshold}_{top_k}_{use_ensemble}_{extractors_str}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        return f"{text_hash}_{params_hash}"

    def get(
        self,
        text: str,
        extractors: List[str] = None,
        threshold: float = 0.2,
        top_k: int = 10,
        use_ensemble: bool = True,
    ) -> Optional[Dict]:
        """
        Get cached extraction results

        Args:
            text: Input text
            extractors: List of extractors
            threshold: Confidence threshold
            top_k: Maximum results
            use_ensemble: Whether ensemble was used

        Returns:
            Cached results or None if not found
        """
        # Build cache key
        key = self._build_cache_key(text, extractors, threshold, top_k, use_ensemble)

        # Try to get from appropriate cache backend
        if self.cache_type == "memory":
            return self._get_from_memory(key)
        elif self.cache_type == "file":
            return self._get_from_file(key)
        elif self.cache_type == "redis":
            return self._get_from_redis(key)

        return None

    def set(
        self,
        text: str,
        result: Dict,
        extractors: List[str] = None,
        threshold: float = 0.2,
        top_k: int = 10,
        use_ensemble: bool = True,
    ) -> None:
        """
        Store extraction results in cache

        Args:
            text: Input text
            result: Extraction results
            extractors: List of extractors
            threshold: Confidence threshold
            top_k: Maximum results
            use_ensemble: Whether ensemble was used
        """
        # Build cache key
        key = self._build_cache_key(text, extractors, threshold, top_k, use_ensemble)

        # Add metadata
        result["cache_metadata"] = {
            "cached_at": datetime.now().isoformat(),
            "ttl": self.ttl,
            "text_hash": self._hash_text(text),
            "extractors": extractors,
        }

        # Store in appropriate cache backend
        if self.cache_type == "memory":
            self._set_in_memory(key, result)
        elif self.cache_type == "file":
            self._set_in_file(key, result)
        elif self.cache_type == "redis":
            self._set_in_redis(key, result)

    def _get_from_memory(self, key: str) -> Optional[Dict]:
        """
        Get results from in-memory cache

        Args:
            key: Cache key

        Returns:
            Cached results or None if not found
        """
        if key not in self.cache:
            return None

        # Check if entry has expired
        timestamp = self.timestamps.get(key, 0)
        if timestamp + self.ttl < time.time():
            # Remove expired entry
            self._cleanup_memory_entry(key)
            return None

        # Update access count
        self.access_counts[key] = self.access_counts.get(key, 0) + 1

        return self.cache[key]

    def _set_in_memory(self, key: str, result: Dict) -> None:
        """
        Store results in in-memory cache

        Args:
            key: Cache key
            result: Extraction results
        """
        # Check if we need to make room
        if len(self.cache) >= self.cache_size_limit:
            self._cleanup_memory_cache()

        # Store result
        self.cache[key] = result
        self.timestamps[key] = time.time()
        self.access_counts[key] = 0

    def _cleanup_memory_cache(self) -> None:
        """Clean up in-memory cache when it reaches size limit"""
        # Strategy: remove least recently accessed entries
        entries = [
            (k, self.timestamps.get(k, 0), self.access_counts.get(k, 0))
            for k in self.cache.keys()
        ]

        # Sort by access count (ascending) and then timestamp (ascending)
        entries.sort(key=lambda x: (x[2], x[1]))

        # Remove oldest entries until we're under the limit
        entries_to_remove = max(1, len(entries) // 10)  # Remove at least 1, up to 10%

        for i in range(min(entries_to_remove, len(entries))):
            self._cleanup_memory_entry(entries[i][0])

    def _cleanup_memory_entry(self, key: str) -> None:
        """Remove a specific entry from memory cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
        if key in self.access_counts:
            del self.access_counts[key]

    def _get_from_file(self, key: str) -> Optional[Dict]:
        """
        Get results from file cache

        Args:
            key: Cache key

        Returns:
            Cached results or None if not found
        """
        file_path = os.path.join(self.cache_dir, f"{key}.json")

        if not os.path.exists(file_path):
            return None

        try:
            # Check if file has expired
            file_mtime = os.path.getmtime(file_path)
            if file_mtime + self.ttl < time.time():
                # Remove expired file
                os.remove(file_path)
                return None

            # Read file
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache file {file_path}: {e}")
            return None

    def _set_in_file(self, key: str, result: Dict) -> None:
        """
        Store results in file cache

        Args:
            key: Cache key
            result: Extraction results
        """
        file_path = os.path.join(self.cache_dir, f"{key}.json")

        try:
            with open(file_path, "w") as f:
                json.dump(result, f)
        except Exception as e:
            logger.error(f"Error writing cache file {file_path}: {e}")

    def _get_from_redis(self, key: str) -> Optional[Dict]:
        """
        Get results from Redis cache

        Args:
            key: Cache key

        Returns:
            Cached results or None if not found
        """
        # Add prefix to avoid key collisions
        redis_key = f"{self.redis_prefix}{key}"

        try:
            value = self.redis_client.get(redis_key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Error getting from Redis: {e}")

        return None

    def _set_in_redis(self, key: str, result: Dict) -> None:
        """
        Store results in Redis cache

        Args:
            key: Cache key
            result: Extraction results
        """
        # Add prefix to avoid key collisions
        redis_key = f"{self.redis_prefix}{key}"

        try:
            self.redis_client.setex(redis_key, self.ttl, json.dumps(result))
        except Exception as e:
            logger.error(f"Error setting in Redis: {e}")

    def invalidate(
        self, text: str = None, extractors: List[str] = None, pattern: str = None
    ) -> int:
        """
        Invalidate cache entries

        Args:
            text: Specific text to invalidate
            extractors: Specific extractors to invalidate
            pattern: Pattern for keys to invalidate

        Returns:
            Number of entries invalidated
        """
        count = 0

        if text is not None:
            # Invalidate specific text
            text_hash = self._hash_text(text)
            count += self._invalidate_pattern(f"{text_hash}_*")
        elif extractors is not None:
            # Invalidate specific extractors (this is approximate)
            extractors_str = ",".join(sorted(extractors))
            extractors_hash = hashlib.md5(extractors_str.encode()).hexdigest()[:8]
            count += self._invalidate_pattern(f"*_{extractors_hash}")
        elif pattern is not None:
            # Invalidate pattern
            count += self._invalidate_pattern(pattern)
        else:
            # Invalidate all
            count += self._invalidate_pattern("*")

        return count

    def _invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern

        Args:
            pattern: Pattern for keys to invalidate

        Returns:
            Number of entries invalidated
        """
        count = 0

        if self.cache_type == "memory":
            # For memory cache, convert glob pattern to regex
            regex_pattern = pattern.replace("*", ".*")
            keys_to_remove = [
                k for k in self.cache.keys() if re.match(regex_pattern, k)
            ]

            for key in keys_to_remove:
                self._cleanup_memory_entry(key)
                count += 1

        elif self.cache_type == "file":
            # For file cache, use glob
            import glob

            file_pattern = os.path.join(self.cache_dir, f"{pattern}.json")
            files = glob.glob(file_pattern)

            for file_path in files:
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Error removing cache file {file_path}: {e}")

        elif self.cache_type == "redis":
            # For Redis, use keys command with pattern
            try:
                redis_pattern = f"{self.redis_prefix}{pattern}"
                keys = self.redis_client.keys(redis_pattern)

                if keys:
                    count = self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Error invalidating Redis keys: {e}")

        return count

    def cleanup(self) -> int:
        """
        Clean up expired cache entries

        Returns:
            Number of entries removed
        """
        count = 0

        if self.cache_type == "memory":
            # For memory cache, check timestamps
            current_time = time.time()
            expired_keys = [
                k for k, t in self.timestamps.items() if t + self.ttl < current_time
            ]

            for key in expired_keys:
                self._cleanup_memory_entry(key)
                count += 1

        elif self.cache_type == "file":
            # For file cache, check file modification times
            current_time = time.time()

            for file_name in os.listdir(self.cache_dir):
                if not file_name.endswith(".json"):
                    continue

                file_path = os.path.join(self.cache_dir, file_name)
                file_mtime = os.path.getmtime(file_path)

                if file_mtime + self.ttl < current_time:
                    try:
                        os.remove(file_path)
                        count += 1
                    except Exception as e:
                        logger.error(
                            f"Error removing expired cache file {file_path}: {e}"
                        )

        # For Redis, expiration is handled automatically

        return count

    def get_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            Dictionary of cache statistics
        """
        stats = {"cache_type": self.cache_type, "ttl": self.ttl}

        if self.cache_type == "memory":
            stats["entries"] = len(self.cache)
            stats["size_limit"] = self.cache_size_limit
            stats["memory_usage_bytes"] = sum(
                len(json.dumps(v)) for v in self.cache.values()
            )

        elif self.cache_type == "file":
            file_count = len(
                [f for f in os.listdir(self.cache_dir) if f.endswith(".json")]
            )
            stats["entries"] = file_count

            # Calculate total size
            total_size = 0
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith(".json"):
                    file_path = os.path.join(self.cache_dir, file_name)
                    total_size += os.path.getsize(file_path)

            stats["disk_usage_bytes"] = total_size

        elif self.cache_type == "redis":
            try:
                # Get stats from Redis
                stats["entries"] = len(self.redis_client.keys(f"{self.redis_prefix}*"))
                info = self.redis_client.info()
                stats["redis_used_memory_bytes"] = info.get("used_memory", 0)
                stats["redis_used_memory_peak_bytes"] = info.get("used_memory_peak", 0)
                stats["redis_version"] = info.get("redis_version", "unknown")
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
                stats["error"] = str(e)

        return stats


class RequestLimiter:
    """
    Rate limiter for API requests based on user tier
    """

    def __init__(
        self,
        redis_url: str = None,
        limits: Dict[str, Dict] = None,
        cache_ttl: int = 3600,  # 1 hour default TTL
    ):
        """
        Initialize request limiter

        Args:
            redis_url: URL for Redis connection (optional, uses in-memory if not provided)
            limits: Dictionary of limits by tier
            cache_ttl: Time-to-live for cache entries in seconds
        """
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl

        # Default limits if not provided
        self.limits = limits or {
            "basic": {
                "requests_per_minute": 30,
                "requests_per_hour": 500,
                "max_text_length": 50000,
                "max_batch_size": 10,
            },
            "premium": {
                "requests_per_minute": 100,
                "requests_per_hour": 2000,
                "max_text_length": 2000000,
                "max_batch_size": 50,
            },
            "enterprise": {
                "requests_per_minute": 300,
                "requests_per_hour": 10000,
                "max_text_length": 5000000,
                "max_batch_size": 200,
            },
        }

        # Initialize Redis client if URL provided
        self.use_redis = False
        self.redis_client = None

        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.use_redis = True
                logger.info(f"Connected to Redis limiter at {redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                logger.info("Falling back to in-memory rate limiting")

        # In-memory counter storage
        self.minute_counters = {}
        self.hour_counters = {}
        self.counter_timestamps = {}

    def check_limit(
        self,
        user_id: str,
        tier: str = "basic",
        text_length: int = 0,
        batch_size: int = 1,
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
        # Get limits for tier
        tier_limits = self.limits.get(tier, self.limits["basic"])

        # Check text length
        max_text_length = tier_limits.get("max_text_length", 50000)
        if text_length > max_text_length:
            return False, {
                "reason": "text_length",
                "limit": max_text_length,
                "requested": text_length,
            }

        # Check batch size
        max_batch_size = tier_limits.get("max_batch_size", 10)
        if batch_size > max_batch_size:
            return False, {
                "reason": "batch_size",
                "limit": max_batch_size,
                "requested": batch_size,
            }

        # Check rate limits
        if self.use_redis:
            return self._check_redis_limit(user_id, tier, tier_limits)
        else:
            return self._check_memory_limit(user_id, tier, tier_limits)

    def _check_redis_limit(
        self, user_id: str, tier: str, tier_limits: Dict
    ) -> Tuple[bool, Dict]:
        """
        Check rate limit using Redis

        Args:
            user_id: User identifier
            tier: User tier
            tier_limits: Limits for this tier

        Returns:
            Tuple of (is_allowed, limit_info)
        """
        current_time = int(time.time())
        minute_key = f"rate_limit:{user_id}:{tier}:minute:{current_time // 60}"
        hour_key = f"rate_limit:{user_id}:{tier}:hour:{current_time // 3600}"

        # Use pipeline for efficiency
        pipe = self.redis_client.pipeline()

        # Increment counters
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)  # Expire after 1 minute
        pipe.incr(hour_key)
        pipe.expire(hour_key, 3600)  # Expire after 1 hour

        # Execute pipeline
        results = pipe.execute()
        minute_count = results[0]
        hour_count = results[2]

        # Check limits
        requests_per_minute = tier_limits.get("requests_per_minute", 30)
        requests_per_hour = tier_limits.get("requests_per_hour", 500)

        if minute_count > requests_per_minute:
            return False, {
                "reason": "rate_limit_minute",
                "limit": requests_per_minute,
                "current": minute_count,
                "reset_in_seconds": 60 - (current_time % 60),
            }

        if hour_count > requests_per_hour:
            return False, {
                "reason": "rate_limit_hour",
                "limit": requests_per_hour,
                "current": hour_count,
                "reset_in_seconds": 3600 - (current_time % 3600),
            }

        return True, {
            "minute_count": minute_count,
            "hour_count": hour_count,
            "minute_limit": requests_per_minute,
            "hour_limit": requests_per_hour,
        }

    def _check_memory_limit(
        self, user_id: str, tier: str, tier_limits: Dict
    ) -> Tuple[bool, Dict]:
        """
        Check rate limit using in-memory counters

        Args:
            user_id: User identifier
            tier: User tier
            tier_limits: Limits for this tier

        Returns:
            Tuple of (is_allowed, limit_info)
        """
        current_time = int(time.time())
        minute_key = f"{user_id}:{tier}:minute:{current_time // 60}"
        hour_key = f"{user_id}:{tier}:hour:{current_time // 3600}"

        # Clean up expired counters
        self._cleanup_memory_counters()

        # Increment minute counter
        if minute_key not in self.minute_counters:
            self.minute_counters[minute_key] = 0
            self.counter_timestamps[minute_key] = current_time

        self.minute_counters[minute_key] += 1
        minute_count = self.minute_counters[minute_key]

        # Increment hour counter
        if hour_key not in self.hour_counters:
            self.hour_counters[hour_key] = 0
            self.counter_timestamps[hour_key] = current_time

        self.hour_counters[hour_key] += 1
        hour_count = self.hour_counters[hour_key]

        # Check limits
        requests_per_minute = tier_limits.get("requests_per_minute", 30)
        requests_per_hour = tier_limits.get("requests_per_hour", 500)

        if minute_count > requests_per_minute:
            return False, {
                "reason": "rate_limit_minute",
                "limit": requests_per_minute,
                "current": minute_count,
                "reset_in_seconds": 60 - (current_time % 60),
            }

        if hour_count > requests_per_hour:
            return False, {
                "reason": "rate_limit_hour",
                "limit": requests_per_hour,
                "current": hour_count,
                "reset_in_seconds": 3600 - (current_time % 3600),
            }

        return True, {
            "minute_count": minute_count,
            "hour_count": hour_count,
            "minute_limit": requests_per_minute,
            "hour_limit": requests_per_hour,
        }

    def _cleanup_memory_counters(self) -> None:
        """Clean up expired in-memory counters"""
        current_time = int(time.time())
        minute_expire = 60
        hour_expire = 3600

        # Remove expired counters
        for key, timestamp in list(self.counter_timestamps.items()):
            if key.endswith("minute") and current_time - timestamp > minute_expire:
                if key in self.minute_counters:
                    del self.minute_counters[key]
                del self.counter_timestamps[key]
            elif key.endswith("hour") and current_time - timestamp > hour_expire:
                if key in self.hour_counters:
                    del self.hour_counters[key]
                del self.counter_timestamps[key]
