"""
Response caching module for reducing API costs and latency.

Implements in-memory caching with TTL support.
"""

import hashlib
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from src.logger import get_logger

logger = get_logger(__name__)


class ResponseCache:
    """
    In-memory cache for RAG responses with TTL support.

    Features:
    - Hash-based cache keys (query + document_id)
    - Time-to-live (TTL) expiration
    - Automatic cleanup of expired entries
    - Cache statistics tracking
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        Initialize response cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 3600 = 1 hour)
            max_size: Maximum number of cache entries before cleanup
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        logger.info(f"ResponseCache initialized (TTL: {ttl_seconds}s, Max Size: {max_size})")

    def _generate_cache_key(self, query: str, document_id: Optional[str] = None,
                           category_filter: Optional[str] = None) -> str:
        """
        Generate cache key from query and optional document/category filters.

        Args:
            query: User query
            document_id: Optional document identifier
            category_filter: Optional category filter

        Returns:
            SHA256 hash of the combined inputs
        """
        # Combine inputs for unique key
        key_string = f"{query}:{document_id or 'all'}:{category_filter or 'all'}"

        # Generate hash
        cache_key = hashlib.sha256(key_string.encode('utf-8')).hexdigest()
        return cache_key

    def get(self, query: str, document_id: Optional[str] = None,
            category_filter: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response if available and not expired.

        Args:
            query: User query
            document_id: Optional document identifier
            category_filter: Optional category filter

        Returns:
            Cached response data or None if not found/expired
        """
        cache_key = self._generate_cache_key(query, document_id, category_filter)

        # Check if key exists
        if cache_key not in self.cache:
            self.misses += 1
            return None

        cached_entry = self.cache[cache_key]
        expiry_time = cached_entry['expiry_time']

        # Check if expired
        if datetime.now() > expiry_time:
            # Remove expired entry
            del self.cache[cache_key]
            self.misses += 1
            logger.debug(f"Cache miss (expired): {cache_key[:16]}...")
            return None

        # Cache hit
        self.hits += 1
        logger.debug(f"Cache hit: {cache_key[:16]}...")
        return cached_entry['response']

    def set(self, query: str, response: Dict[str, Any],
            document_id: Optional[str] = None,
            category_filter: Optional[str] = None) -> None:
        """
        Store response in cache with TTL.

        Args:
            query: User query
            response: Response data to cache
            document_id: Optional document identifier
            category_filter: Optional category filter
        """
        cache_key = self._generate_cache_key(query, document_id, category_filter)

        # Check if cache is full and needs cleanup
        if len(self.cache) >= self.max_size:
            self._cleanup_expired()

            # If still full after cleanup, remove oldest entry
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(),
                               key=lambda k: self.cache[k]['created_at'])
                del self.cache[oldest_key]
                self.evictions += 1
                logger.debug(f"Cache eviction (full): {oldest_key[:16]}...")

        # Store with expiry time
        self.cache[cache_key] = {
            'response': response,
            'created_at': datetime.now(),
            'expiry_time': datetime.now() + timedelta(seconds=self.ttl_seconds),
            'query': query[:100],  # Store truncated query for debugging
        }

        logger.debug(f"Cache set: {cache_key[:16]}...")

    def _cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = [
            key for key, value in self.cache.items()
            if value['expiry_time'] < now
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cache cleared ({count} entries removed)")

    def invalidate_document(self, document_id: str) -> int:
        """
        Invalidate all cache entries for a specific document.

        Args:
            document_id: Document identifier

        Returns:
            Number of entries invalidated
        """
        # This is a simplified implementation
        # In production, you'd want to track document_id mappings
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Invalidated cache for document: {document_id}")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'total_requests': total_requests,
            'hit_rate': f"{hit_rate:.2f}%",
            'current_size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        logger.info("Cache statistics reset")


# Singleton instance
_cache_instance: Optional[ResponseCache] = None


def get_cache(ttl_seconds: int = 3600, max_size: int = 1000) -> ResponseCache:
    """
    Get or create singleton cache instance.

    Args:
        ttl_seconds: Time-to-live for cache entries
        max_size: Maximum cache size

    Returns:
        ResponseCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResponseCache(ttl_seconds=ttl_seconds, max_size=max_size)
    return _cache_instance
