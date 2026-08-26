"""
Caching Service - In-Memory Cache for Performance
Caches generated posts, vector searches, and API responses.
"""

import logging
import time
import hashlib
from typing import Any, Optional, Dict, Callable
from functools import wraps
from collections import OrderedDict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LRUCache:
    """
    Simple LRU (Least Recently Used) cache with TTL support.
    Thread-safe for basic operations.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds (5 min default)
        """
        self.cache = OrderedDict()
        self.timestamps = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0

        logger.info(f"✅ Cache initialized (max_size={max_size}, ttl={default_ttl}s)")

    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry has expired."""
        if key not in self.timestamps:
            return True

        timestamp, ttl = self.timestamps[key]
        return time.time() - timestamp > ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self.misses += 1
            return None

        if self._is_expired(key):
            # Remove expired entry
            del self.cache[key]
            del self.timestamps[key]
            self.misses += 1
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1

        return self.cache[key]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove oldest (LRU)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]

        self.cache[key] = value
        self.timestamps[key] = (time.time(), ttl or self.default_ttl)

    def delete(self, key: str) -> None:
        """Delete a key from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]

    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
        self.timestamps.clear()
        self.hits = 0
        self.misses = 0
        logger.info("🗑️ Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "total_requests": total_requests
        }


# Global cache instances
post_cache = LRUCache(max_size=500, default_ttl=300)  # 5 min TTL
vector_cache = LRUCache(max_size=1000, default_ttl=600)  # 10 min TTL
image_cache = LRUCache(max_size=200, default_ttl=1800)  # 30 min TTL
api_cache = LRUCache(max_size=100, default_ttl=60)  # 1 min TTL


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from function arguments."""
    # Combine args and kwargs into a single string
    key_parts = []

    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            # For complex objects, use their string representation
            key_parts.append(str(arg))

    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}={v}")
        else:
            key_parts.append(f"{k}={str(v)}")

    # Hash the combined key for consistent length
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(cache_instance: LRUCache = post_cache, ttl: Optional[int] = None):
    """
    Decorator to cache function results.

    Usage:
        @cached(post_cache, ttl=300)
        async def generate_post(...):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{generate_cache_key(*args, **kwargs)}"

            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                logger.info(f"✅ Cache HIT: {func.__name__}")
                return cached_result

            # Cache miss - call function
            logger.info(f"❌ Cache MISS: {func.__name__}")
            result = await func(*args, **kwargs)

            # Store in cache
            cache_instance.set(cache_key, result, ttl)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{generate_cache_key(*args, **kwargs)}"

            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                logger.info(f"✅ Cache HIT: {func.__name__}")
                return cached_result

            # Cache miss - call function
            logger.info(f"❌ Cache MISS: {func.__name__}")
            result = func(*args, **kwargs)

            # Store in cache
            cache_instance.set(cache_key, result, ttl)

            return result

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def cache_post_generation(user_id: str, transcript: str, platform: str, tone: str, result: Any) -> None:
    """Manually cache a post generation result."""
    cache_key = f"post:{user_id}:{hashlib.md5(f'{transcript}{platform}{tone}'.encode()).hexdigest()}"
    post_cache.set(cache_key, result, ttl=300)
    logger.info(f"📦 Cached post generation for user {user_id}")


def get_cached_post(user_id: str, transcript: str, platform: str, tone: str) -> Optional[Any]:
    """Get cached post generation result."""
    cache_key = f"post:{user_id}:{hashlib.md5(f'{transcript}{platform}{tone}'.encode()).hexdigest()}"
    return post_cache.get(cache_key)


def cache_vector_search(query: str, user_id: str, result: Any) -> None:
    """Cache a vector search result."""
    cache_key = f"vector:{user_id}:{hashlib.md5(query.encode()).hexdigest()}"
    vector_cache.set(cache_key, result, ttl=600)


def get_cached_vector_search(query: str, user_id: str) -> Optional[Any]:
    """Get cached vector search result."""
    cache_key = f"vector:{user_id}:{hashlib.md5(query.encode()).hexdigest()}"
    return vector_cache.get(cache_key)


def cache_image(post_text: str, platform: str, result: Any) -> None:
    """Cache an image generation result."""
    cache_key = f"image:{platform}:{hashlib.md5(post_text.encode()).hexdigest()}"
    image_cache.set(cache_key, result, ttl=1800)


def get_cached_image(post_text: str, platform: str) -> Optional[Any]:
    """Get cached image result."""
    cache_key = f"image:{platform}:{hashlib.md5(post_text.encode()).hexdigest()}"
    return image_cache.get(cache_key)


def invalidate_user_cache(user_id: str) -> None:
    """Invalidate all cache entries for a specific user."""
    # Clear post cache entries for user
    keys_to_delete = [k for k in post_cache.cache.keys() if user_id in k]
    for key in keys_to_delete:
        post_cache.delete(key)

    # Clear vector cache entries for user
    keys_to_delete = [k for k in vector_cache.cache.keys() if user_id in k]
    for key in keys_to_delete:
        vector_cache.delete(key)

    logger.info(f"🗑️ Invalidated cache for user {user_id}")


def get_all_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches."""
    return {
        "post_cache": post_cache.stats(),
        "vector_cache": vector_cache.stats(),
        "image_cache": image_cache.stats(),
        "api_cache": api_cache.stats(),
        "total_memory_items": (
            len(post_cache.cache) +
            len(vector_cache.cache) +
            len(image_cache.cache) +
            len(api_cache.cache)
        )
    }


def clear_all_caches() -> None:
    """Clear all caches."""
    post_cache.clear()
    vector_cache.clear()
    image_cache.clear()
    api_cache.clear()
    logger.info("🗑️ All caches cleared")


# Warmup function to pre-populate cache
def warmup_cache():
    """Pre-populate cache with common queries (optional)."""
    logger.info("🔥 Cache warmup skipped (no common queries defined)")
    pass


# Initialize
logger.info("✅ Caching service initialized")
logger.info(f"📊 Cache instances: post({post_cache.max_size}), vector({vector_cache.max_size}), image({image_cache.max_size}), api({api_cache.max_size})")
