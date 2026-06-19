"""
pipeline/image_validator.py
────────────────────────────
Layer 2 — OpenCV image quality pre-pass.

Runs fast, zero-cost image quality checks before any LLM API call.
Flags blurry, dark, overexposed, or low-complexity images.
These pre-computed flags are injected into the VLM prompt as context.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageQualityResult:
    """Result of quality analysis for a single image."""

    def __init__(
        self,
        path: str,
        valid: bool,
        flags: List[str],
        blur_score: float = 0.0,
        brightness: float = 127.0,
        entropy: float = 7.0,
        width: int = 0,
        height: int = 0,
        error: str = "",
    ):
        self.path = path
        self.image_id = Path(path).stem  # e.g. img_1
        self.valid = valid
        self.flags = flags
        self.blur_score = blur_score
        self.brightness = brightness
        self.entropy = entropy
        self.width = width
        self.height = height
        self.error = error

    def __repr__(self) -> str:
        return (
            f"ImageQualityResult(id={self.image_id}, valid={self.valid}, "
            f"flags={self.flags}, blur={self.blur_score:.1f})"
        )


class ImageValidator:
    """
    Runs physical image quality checks using OpenCV.

    Checks performed:
    1. Laplacian variance for blur detection
    2. Mean grayscale brightness for exposure
    3. Color entropy for information content
    4. Minimum resolution check
    5. Edge density for obstruction detection
    """

    def __init__(
        self,
        blur_threshold: float = 45.0,
        brightness_min: float = 50.0,
        brightness_max: float = 210.0,
        entropy_threshold: float = 3.0,
        min_width: int = 200,
        min_height: int = 200,
    ):
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.entropy_threshold = entropy_threshold
        self.min_width = min_width
        self.min_height = min_height

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_image(self, image_path: str) -> ImageQualityResult:
        """Validate a single image and return quality flags."""
        from utils.image_hash_cache import IMAGE_CACHE

        # Check deduplication cache first
        cached = IMAGE_CACHE.get(image_path)
        if cached is not None:
            return cached  # type: ignore[return-value]

        path = str(image_path)

        # Load image
        img = cv2.imread(path)
        if img is None:
            logger.warning("Could not load image: %s", path)
            return ImageQualityResult(
                path=path,
                valid=False,
                flags=["damage_not_visible"],
                error="Image could not be loaded",
            )

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flags: List[str] = []
        is_valid = True  # Only set False for blur (skips VLM call)

        # ---- 1. Blur detection (Laplacian variance) ----
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.blur_threshold:
            flags.append("blurry_image")
            is_valid = False  # Blurry → skip VLM, can't evaluate

        # ---- 2. Brightness / exposure ----
        brightness = float(np.mean(gray))
        if brightness < self.brightness_min or brightness > self.brightness_max:
            flags.append("low_light_or_glare")
            # Allow VLM call but with reduced confidence

        # ---- 3. Resolution check ----
        if h < self.min_height or w < self.min_width:
            if "damage_not_visible" not in flags:
                flags.append("damage_not_visible")

        # ---- 4. Color entropy (information content) ----
        entropy = self._compute_entropy(gray)
        if entropy < self.entropy_threshold:
            if "cropped_or_obstructed" not in flags:
                flags.append("cropped_or_obstructed")

        # ---- 5. Edge density (obstruction / cropping) ----
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / (h * w) if h * w > 0 else 0
        if edge_ratio > 0.95:
            if "cropped_or_obstructed" not in flags:
                flags.append("cropped_or_obstructed")

        result = ImageQualityResult(
            path=path,
            valid=is_valid,
            flags=flags if flags else [],
            blur_score=blur_score,
            brightness=brightness,
            entropy=entropy,
            width=w,
            height=h,
        )
        
        IMAGE_CACHE.put(image_path, result)
        return result

    def validate_image_set(
        self, image_paths: List[str]
    ) -> Tuple[bool, List[str], List[ImageQualityResult]]:
        """
        Validate all images for a claim.

        Returns:
            overall_valid: True if at least one image is usable
            aggregated_flags: merged unique flags across all images
            results: per-image quality results
        """
        results: List[ImageQualityResult] = []
        all_flags: List[str] = []

        for path in image_paths:
            result = self.validate_image(path)
            results.append(result)
            all_flags.extend(result.flags)

        # De-duplicate while preserving order
        seen: Dict[str, bool] = {}
        unique_flags = [f for f in all_flags if not (f in seen or seen.update({f: True}))]  # type: ignore

        # Overall validity: at least one image must be non-blurry
        overall_valid = any(r.valid for r in results)

        return overall_valid, unique_flags, results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entropy(gray: np.ndarray) -> float:
        """
        Compute Shannon entropy of the grayscale histogram.
        H = -sum(p(i) * log2(p(i)))
        Low entropy → blank or featureless image.
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        total = hist.sum()
        if total == 0:
            return 0.0

        probs = hist / total
        # Filter zero-probability bins to avoid log(0)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
        return entropy
