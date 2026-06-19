#!/usr/bin/env python3
"""
utils/image_hash_cache.py
──────────────────────────
SHA-256-based image deduplication cache.

Identical images submitted across multiple claims share OpenCV
analysis results in memory, avoiding redundant computation.
This is especially useful for batch runs over the test set where
users may submit duplicate or similar images.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ImageHashCache:
    """
    In-memory cache for image analysis results keyed by SHA-256 hash.
    Prevents duplicate OpenCV computation across claims.
    """

    def __init__(self):
        self._hash_to_result: Dict[str, object] = {}
        self._path_to_hash: Dict[str, str] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(self, image_path: str) -> Optional[object]:
        """Return cached result for this image path, or None."""
        file_hash = self._get_or_compute_hash(image_path)
        if file_hash and file_hash in self._hash_to_result:
            self.hits += 1
            logger.debug("Cache HIT for %s (hash=%s…)", Path(image_path).name, file_hash[:8])
            return self._hash_to_result[file_hash]
        self.misses += 1
        return None

    def put(self, image_path: str, result: object) -> None:
        """Store result for this image path."""
        file_hash = self._get_or_compute_hash(image_path)
        if file_hash:
            self._hash_to_result[file_hash] = result
            logger.debug("Cache SET for %s (hash=%s…)", Path(image_path).name, file_hash[:8])

    def _get_or_compute_hash(self, image_path: str) -> Optional[str]:
        """Compute SHA-256 hash of image file, caching the hash itself."""
        if image_path in self._path_to_hash:
            return self._path_to_hash[image_path]

        try:
            path = Path(image_path)
            if not path.exists():
                return None
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self._path_to_hash[image_path] = sha256
            return sha256
        except Exception as e:
            logger.warning("Could not hash %s: %s", image_path, e)
            return None

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def log_stats(self) -> None:
        logger.info(
            "Image cache stats — hits=%d, misses=%d, hit_rate=%.1f%%",
            self.hits, self.misses, self.hit_rate * 100
        )


# Global singleton (shared across pipeline runs)
IMAGE_CACHE = ImageHashCache()
