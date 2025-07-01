# src/api/v2/routes/Optimized_Cache.py
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import redis
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class TieredCache:
    """
    Multi-level caching system with tiered expiration and intelligent invalidation
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        memory_cache_size: int = 1000,
        memory_ttl: int = 300,  # 5 minutes
        redis_ttl: int = 86400,  # 1 day
        file_cache_dir: str = "cache",
        file_ttl: int = 604800,  # 1 week
        enable_request_cache: bool = True,
        enable_memory_cache: bool = True,
        enable_redis_cache: bool = True,
        enable_file_cache: bool = True,
    ):
        """
        Initialize tiered cache

        Args:
            redis_url: Redis connection URL
            memory_cache_size: Maximum number of items in memory cache
            memory_ttl: Memory cache TTL in seconds
            redis_ttl: Redis cache TTL in seconds
            file_cache_dir: Directory for file cache
            file_ttl: File cache TTL in seconds
            enable_request_cache: Whether to enable request-level cache
            enable_memory_cache: Whether to enable in-memory cache
            enable_redis_cache: Whether to enable Redis cache
            enable_file_cache: Whether to enable file cache
        """
        self.memory_cache_size = memory_cache_size
        self.memory_ttl = memory_ttl
        self.redis_ttl = redis_ttl
        self.file_cache_dir = file_cache_dir
        self.file_ttl = file_ttl

        # Enable/disable specific cache levels
        self.enable_request_cache = enable_request_cache
        self.enable_memory_cache = enable_memory_cache
        self.enable_redis_cache = enable_redis_cache and redis_url is not None
        self.enable_file_cache = enable_file_cache

        # Initialize memory cache
        self.memory_cache = {}
        self.memory_cache_access_times = {}

        # Initialize Redis connection if enabled
        self.redis_client = None
        if self.enable_redis_cache:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info(f"Connected to Redis cache at {redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.enable_redis_cache = False

        # Create file cache directory if enabled
        if self.enable_file_cache:
            os.makedirs(self.file_cache_dir, exist_ok=True)

        # Cache stats
        self.stats = {
            "hits": {"request": 0, "memory": 0, "redis": 0, "file": 0, "total": 0},
            "misses": {"total": 0},
            "sets": {"memory": 0, "redis": 0, "file": 0, "total": 0},
            "evictions": 0,
            "errors": 0,
            "start_time": time.time(),
        }

        logger.info(
            f"Initialized tiered cache (memory={enable_memory_cache}, redis={enable_redis_cache}, file={enable_file_cache})"
        )

    def _generate_key(self, key_parts: Any) -> str:
        """
        Generate cache key from parts

        Args:
            key_parts: Parts to include in key generation

        Returns:
            Cache key string
        """
        # Convert key parts to string representation
        key_str = json.dumps(key_parts, sort_keys=True)

        # Generate hash
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key_parts: Any) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key_parts: Key parts to look up

        Returns:
            Cached value or None if not found
        """
        cache_key = self._generate_key(key_parts)

        # Check request cache first (if implemented externally)
        # This would be handled by FastAPI middleware

        # Check memory cache
        if self.enable_memory_cache:
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]

                # Check if entry is expired
                if entry["expiry"] > time.time():
                    # Update access time
                    self.memory_cache_access_times[cache_key] = time.time()

                    # Update stats
                    self.stats["hits"]["memory"] += 1
                    self.stats["hits"]["total"] += 1

                    return entry["value"]
                else:
                    # Remove expired entry
                    del self.memory_cache[cache_key]
                    del self.memory_cache_access_times[cache_key]

        # Check Redis cache
        if self.enable_redis_cache and self.redis_client:
            try:
                redis_key = f"cache:{cache_key}"
                cached_data = self.redis_client.get(redis_key)

                if cached_data:
                    # Deserialize
                    value = json.loads(cached_data)

                    # Update memory cache
                    if self.enable_memory_cache:
                        self._set_memory_cache(cache_key, value)

                    # Update stats
                    self.stats["hits"]["redis"] += 1
                    self.stats["hits"]["total"] += 1

                    return value
            except Exception as e:
                logger.error(f"Redis cache error: {e}")
                self.stats["errors"] += 1

        # Check file cache
        if self.enable_file_cache:
            file_path = os.path.join(self.file_cache_dir, f"{cache_key}.json")

            if os.path.exists(file_path):
                try:
                    # Check if file is expired
                    file_mtime = os.path.getmtime(file_path)
                    if time.time() - file_mtime <= self.file_ttl:
                        # Read file
                        with open(file_path, "r") as f:
                            value = json.load(f)

                        # Update memory and Redis caches
                        if self.enable_memory_cache:
                            self._set_memory_cache(cache_key, value)

                        if self.enable_redis_cache and self.redis_client:
                            self._set_redis_cache(cache_key, value)

                        # Update stats
                        self.stats["hits"]["file"] += 1
                        self.stats["hits"]["total"] += 1

                        return value
                    else:
                        # Remove expired file
                        os.remove(file_path)
                except Exception as e:
                    logger.error(f"File cache error: {e}")
                    self.stats["errors"] += 1

                    # Try to remove corrupted file
                    try:
                        os.remove(file_path)
                    except:
                        pass

        # Cache miss
        self.stats["misses"]["total"] += 1
        return None

    def set(self, key_parts: Any, value: Any) -> None:
        """
        Set value in cache

        Args:
            key_parts: Key parts to store under
            value: Value to cache
        """
        cache_key = self._generate_key(key_parts)

        # Update memory cache
        if self.enable_memory_cache:
            self._set_memory_cache(cache_key, value)

        # Update Redis cache
        if self.enable_redis_cache and self.redis_client:
            self._set_redis_cache(cache_key, value)

        # Update file cache
        if self.enable_file_cache:
            self._set_file_cache(cache_key, value)

        # Update stats
        self.stats["sets"]["total"] += 1

    def _set_memory_cache(self, cache_key: str, value: Any) -> None:
        """
        Set value in memory cache

        Args:
            cache_key: Cache key
            value: Value to cache
        """
        # Check if we need to evict entries
        if len(self.memory_cache) >= self.memory_cache_size:
            self._evict_memory_cache_entries()

        # Calculate expiry
        expiry = time.time() + self.memory_ttl

        # Store in memory cache
        self.memory_cache[cache_key] = {"value": value, "expiry": expiry}
        self.memory_cache_access_times[cache_key] = time.time()

        # Update stats
        self.stats["sets"]["memory"] += 1

    def _set_redis_cache(self, cache_key: str, value: Any) -> None:
        """
        Set value in Redis cache

        Args:
            cache_key: Cache key
            value: Value to cache
        """
        try:
            # Serialize value
            serialized = json.dumps(value)

            # Store in Redis with TTL
            redis_key = f"cache:{cache_key}"
            self.redis_client.setex(redis_key, self.redis_ttl, serialized)

            # Update stats
            self.stats["sets"]["redis"] += 1
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            self.stats["errors"] += 1

    def _set_file_cache(self, cache_key: str, value: Any) -> None:
        """
        Set value in file cache

        Args:
            cache_key: Cache key
            value: Value to cache
        """
        try:
            # Write to file
            file_path = os.path.join(self.file_cache_dir, f"{cache_key}.json")
            with open(file_path, "w") as f:
                json.dump(value, f)

            # Update stats
            self.stats["sets"]["file"] += 1
        except Exception as e:
            logger.error(f"File cache error: {e}")
            self.stats["errors"] += 1

    def _evict_memory_cache_entries(self) -> None:
        """Evict least recently used entries from memory cache"""
        # Sort entries by access time
        entries = sorted(self.memory_cache_access_times.items(), key=lambda x: x[1])

        # Remove oldest entries to bring cache size down to 75% of max
        entries_to_remove = max(1, int(0.25 * self.memory_cache_size))

        for i in range(entries_to_remove):
            if i < len(entries):
                key = entries[i][0]
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if key in self.memory_cache_access_times:
                    del self.memory_cache_access_times[key]

                # Update stats
                self.stats["evictions"] += 1

    def invalidate(self, key_parts: Any = None, pattern: str = None) -> int:
        """
        Invalidate cache entries

        Args:
            key_parts: Specific key parts to invalidate
            pattern: Pattern to match for invalidation

        Returns:
            Number of entries invalidated
        """
        invalidated = 0

        if key_parts is not None:
            # Invalidate specific key
            cache_key = self._generate_key(key_parts)

            # Remove from memory cache
            if self.enable_memory_cache:
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                    if cache_key in self.memory_cache_access_times:
                        del self.memory_cache_access_times[cache_key]
                    invalidated += 1

            # Remove from Redis cache
            if self.enable_redis_cache and self.redis_client:
                try:
                    redis_key = f"cache:{cache_key}"
                    deleted = self.redis_client.delete(redis_key)
                    invalidated += deleted
                except Exception as e:
                    logger.error(f"Redis invalidation error: {e}")
                    self.stats["errors"] += 1

            # Remove from file cache
            if self.enable_file_cache:
                file_path = os.path.join(self.file_cache_dir, f"{cache_key}.json")
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        invalidated += 1
                    except Exception as e:
                        logger.error(f"File invalidation error: {e}")
                        self.stats["errors"] += 1

        elif pattern is not None:
            # Invalidate by pattern

            # Memory cache - search for matching keys
            if self.enable_memory_cache:
                keys_to_remove = []
                for key in self.memory_cache.keys():
                    if pattern in key:
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    if key in self.memory_cache:
                        del self.memory_cache[key]
                    if key in self.memory_cache_access_times:
                        del self.memory_cache_access_times[key]

                invalidated += len(keys_to_remove)

            # Redis cache - use scan and delete
            if self.enable_redis_cache and self.redis_client:
                try:
                    pattern_key = f"cache:*{pattern}*"
                    cursor = 0
                    while True:
                        cursor, keys = self.redis_client.scan(
                            cursor, match=pattern_key, count=100
                        )
                        if keys:
                            deleted = self.redis_client.delete(*keys)
                            invalidated += deleted

                        if cursor == 0:
                            break
                except Exception as e:
                    logger.error(f"Redis pattern invalidation error: {e}")
                    self.stats["errors"] += 1

            # File cache - scan directory for matching files
            if self.enable_file_cache:
                try:
                    for filename in os.listdir(self.file_cache_dir):
                        if pattern in filename and filename.endswith(".json"):
                            file_path = os.path.join(self.file_cache_dir, filename)
                            try:
                                os.remove(file_path)
                                invalidated += 1
                            except:
                                pass
                except Exception as e:
                    logger.error(f"File pattern invalidation error: {e}")
                    self.stats["errors"] += 1

        return invalidated

    def clear(self) -> None:
        """Clear all cache entries"""
        # Clear memory cache
        if self.enable_memory_cache:
            self.memory_cache.clear()
            self.memory_cache_access_times.clear()

        # Clear Redis cache
        if self.enable_redis_cache and self.redis_client:
            try:
                # Delete all keys with prefix
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(
                        cursor, match="cache:*", count=100
                    )
                    if keys:
                        self.redis_client.delete(*keys)

                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
                self.stats["errors"] += 1

        # Clear file cache
        if self.enable_file_cache:
            try:
                for filename in os.listdir(self.file_cache_dir):
                    if filename.endswith(".json"):
                        file_path = os.path.join(self.file_cache_dir, filename)
                        try:
                            os.remove(file_path)
                        except:
                            pass
            except Exception as e:
                logger.error(f"File clear error: {e}")
                self.stats["errors"] += 1

        # Reset stats
        self.stats = {
            "hits": {"request": 0, "memory": 0, "redis": 0, "file": 0, "total": 0},
            "misses": {"total": 0},
            "sets": {"memory": 0, "redis": 0, "file": 0, "total": 0},
            "evictions": 0,
            "errors": 0,
            "start_time": time.time(),
        }

    def cleanup(self) -> int:
        """
        Clean up expired entries

        Returns:
            Number of entries removed
        """
        removed = 0

        # Clean memory cache
        if self.enable_memory_cache:
            current_time = time.time()
            keys_to_remove = []

            for key, entry in self.memory_cache.items():
                if entry["expiry"] <= current_time:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.memory_cache[key]
                if key in self.memory_cache_access_times:
                    del self.memory_cache_access_times[key]

            removed += len(keys_to_remove)

        # Clean file cache
        if self.enable_file_cache:
            current_time = time.time()

            try:
                for filename in os.listdir(self.file_cache_dir):
                    if filename.endswith(".json"):
                        file_path = os.path.join(self.file_cache_dir, filename)

                        # Check if file is expired
                        file_mtime = os.path.getmtime(file_path)
                        if current_time - file_mtime > self.file_ttl:
                            try:
                                os.remove(file_path)
                                removed += 1
                            except:
                                pass
            except Exception as e:
                logger.error(f"File cleanup error: {e}")
                self.stats["errors"] += 1

        return removed

    def get_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            Dictionary of cache statistics
        """
        # Calculate additional stats
        total_requests = self.stats["hits"]["total"] + self.stats["misses"]["total"]
        hit_rate = 0
        if total_requests > 0:
            hit_rate = self.stats["hits"]["total"] / total_requests

        uptime = time.time() - self.stats["start_time"]

        # Memory cache size
        memory_size = len(self.memory_cache)

        # Redis cache size (if available)
        redis_size = None
        if self.enable_redis_cache and self.redis_client:
            try:
                redis_size = self.redis_client.dbsize()
            except:
                pass

        # File cache size
        file_size = 0
        if self.enable_file_cache:
            try:
                file_count = len(
                    [f for f in os.listdir(self.file_cache_dir) if f.endswith(".json")]
                )
                file_size = file_count
            except:
                pass

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "evictions": self.stats["evictions"],
            "errors": self.stats["errors"],
            "uptime_seconds": uptime,
            "cache_size": {
                "memory": memory_size,
                "redis": redis_size,
                "file": file_size,
            },
            "enabled": {
                "memory": self.enable_memory_cache,
                "redis": self.enable_redis_cache,
                "file": self.enable_file_cache,
                "request": self.enable_request_cache,
            },
            "ttl": {
                "memory": self.memory_ttl,
                "redis": self.redis_ttl,
                "file": self.file_ttl,
            },
        }


import json

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware for request-level caching"""

    def __init__(self, app, cache):
        """
        Initialize middleware

        Args:
            app: FastAPI application
            cache: Cache instance
        """
        super().__init__(app)
        self.cache = cache
        self.cache_enabled_routes = {
            "GET": ["/api/v1/health", "/api/v1/extract", "/api/v2/extract"],
            "POST": ["/api/v1/extract", "/api/v2/extract"],
        }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process request with caching

        Args:
            request: Request object
            call_next: Next middleware function

        Returns:
            Response object
        """
        # Check if caching is enabled for this route
        method = request.method
        path = request.url.path

        if method in self.cache_enabled_routes and any(
            path.startswith(route) for route in self.cache_enabled_routes[method]
        ):
            # Generate cache key
            cache_key = await self._generate_cache_key(request)

            # Try to get from cache
            cached_response = self.cache.get(cache_key)

            if cached_response is not None:
                # Return cached response
                self.cache.stats["hits"]["request"] += 1
                return JSONResponse(
                    content=cached_response["content"],
                    status_code=cached_response["status_code"],
                    headers={"X-Cache": "HIT"},
                )

            # Process request
            response = await call_next(request)

            # Cache response if it's successful
            if response.status_code == 200:
                try:
                    # Get response body
                    body = await response.body()
                    content = json.loads(body)

                    # Cache response
                    self.cache.set(
                        cache_key,
                        {"content": content, "status_code": response.status_code},
                    )

                    # Return new response with cache header
                    return JSONResponse(
                        content=content,
                        status_code=response.status_code,
                        headers={"X-Cache": "MISS"},
                    )
                except:
                    # If we can't process the response, just return it as-is
                    pass

            return response
        else:
            # Bypass cache for non-cacheable routes
            return await call_next(request)

    async def _generate_cache_key(self, request: Request) -> Dict:
        """
        Generate cache key from request

        Args:
            request: Request object

        Returns:
            Cache key dictionary
        """
        # Get method and path
        method = request.method
        path = request.url.path

        # Get query parameters
        query_params = dict(request.query_params)

        # Get request body for POST requests
        body = None
        if method == "POST":
            try:
                body = await request.json()
            except:
                body = {}

        # Extract user ID or API key
        auth_data = {}
        try:
            # Get user ID from state if available
            if hasattr(request.state, "user"):
                auth_data["user_id"] = request.state.user.get("user_id")

            # Get API key from headers
            api_key = request.headers.get("X-API-Key")
            if api_key:
                auth_data["api_key"] = api_key
        except:
            pass

        # Combine into cache key
        return {
            "method": method,
            "path": path,
            "query": query_params,
            "body": body,
            "auth": auth_data,
        }
