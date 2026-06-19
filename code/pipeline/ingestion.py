"""
pipeline/ingestion.py
─────────────────────
Layer 1 — Data ingestion and context enrichment.

Loads claims.csv, user_history.csv, and evidence_requirements.csv,
then joins them into a unified ClaimContext object for each row.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UserHistory:
    """Parsed user history profile."""
    user_id: str
    past_claim_count: int = 0
    accept_claim: int = 0
    manual_review_claim: int = 0
    rejected_claim: int = 0
    last_90_days_claim_count: int = 0
    history_flags: str = "none"
    history_summary: str = ""

    @property
    def rejection_ratio(self) -> float:
        """R_rej = rejected_claim / past_claim_count"""
        if self.past_claim_count == 0:
            return 0.0
        return self.rejected_claim / self.past_claim_count


@dataclass
class EvidenceRequirement:
    """A single evidence requirement rule."""
    requirement_id: str
    claim_object: str        # car | laptop | package | all
    applies_to: str          # issue family description
    minimum_image_evidence: str


@dataclass
class ClaimContext:
    """
    Fully enriched context for one claim row.
    This is the unified input to Layer 2 (image validation) and
    Layer 3 (VLM reasoning).
    """
    # Raw input fields
    user_id: str
    image_paths: str           # semicolon-separated original paths
    user_claim: str
    claim_object: str          # car | laptop | package

    # Derived fields
    image_path_list: List[str] = field(default_factory=list)  # absolute paths
    image_ids: List[str] = field(default_factory=list)        # img_1, img_2 …

    # Joined data
    user_history: Optional[UserHistory] = None
    applicable_requirements: List[EvidenceRequirement] = field(default_factory=list)

    # Pre-computed risk flags (populated by risk_scorer)
    computed_risk_flags: List[str] = field(default_factory=list)

    # Image quality flags from OpenCV layer (populated by image_validator)
    image_quality_flags: List[str] = field(default_factory=list)
    valid_images_count: int = 0


# ---------------------------------------------------------------------------
# Ingestion engine
# ---------------------------------------------------------------------------

class DataIngestionEngine:
    """
    Loads and joins all CSV data sources.
    Call load_all() once, then get_claim_contexts() to iterate.
    """

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self._claims_df: Optional[pd.DataFrame] = None
        self._history_map: Dict[str, UserHistory] = {}
        self._requirements: List[EvidenceRequirement] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load all CSVs from dataset_dir."""
        self._load_claims()
        self._load_user_history()
        self._load_evidence_requirements()
        logger.info(
            "Ingestion complete — %d claims, %d users, %d evidence rules",
            len(self._claims_df),
            len(self._history_map),
            len(self._requirements),
        )

    def load_sample(self) -> None:
        """Load sample_claims.csv (input + expected output columns)."""
        path = self.dataset_dir / "sample_claims.csv"
        self._claims_df = pd.read_csv(path, dtype=str).fillna("")
        self._load_user_history()
        self._load_evidence_requirements()
        logger.info("Sample dataset loaded — %d rows", len(self._claims_df))

    def get_claim_contexts(self) -> List[ClaimContext]:
        """Return a list of ClaimContext objects from the loaded claims."""
        if self._claims_df is None:
            raise RuntimeError("Call load_all() or load_sample() first.")

        contexts: List[ClaimContext] = []
        for _, row in self._claims_df.iterrows():
            ctx = self._build_context(row)
            contexts.append(ctx)
        return contexts

    def get_sample_ground_truth(self) -> pd.DataFrame:
        """Return sample_claims.csv as a DataFrame (for evaluation)."""
        if self._claims_df is None:
            raise RuntimeError("Call load_sample() first.")
        return self._claims_df.copy()

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_claims(self) -> None:
        path = self.dataset_dir / "claims.csv"
        self._claims_df = pd.read_csv(path, dtype=str).fillna("")
        logger.debug("Loaded %d rows from %s", len(self._claims_df), path)

    def _load_user_history(self) -> None:
        path = self.dataset_dir / "user_history.csv"
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, row in df.iterrows():
            uid = row["user_id"].strip()
            self._history_map[uid] = UserHistory(
                user_id=uid,
                past_claim_count=int(row.get("past_claim_count", 0) or 0),
                accept_claim=int(row.get("accept_claim", 0) or 0),
                manual_review_claim=int(row.get("manual_review_claim", 0) or 0),
                rejected_claim=int(row.get("rejected_claim", 0) or 0),
                last_90_days_claim_count=int(row.get("last_90_days_claim_count", 0) or 0),
                history_flags=row.get("history_flags", "none").strip(),
                history_summary=row.get("history_summary", "").strip(),
            )
        logger.debug("Loaded %d user history records", len(self._history_map))

    def _load_evidence_requirements(self) -> None:
        path = self.dataset_dir / "evidence_requirements.csv"
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, row in df.iterrows():
            self._requirements.append(EvidenceRequirement(
                requirement_id=row["requirement_id"].strip(),
                claim_object=row["claim_object"].strip().lower(),
                applies_to=row["applies_to"].strip(),
                minimum_image_evidence=row["minimum_image_evidence"].strip(),
            ))
        logger.debug("Loaded %d evidence requirements", len(self._requirements))

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self, row: pd.Series) -> ClaimContext:
        user_id = str(row["user_id"]).strip()
        image_paths_raw = str(row["image_paths"]).strip()
        user_claim = str(row["user_claim"]).strip()
        claim_object = str(row["claim_object"]).strip().lower()

        # Parse image paths and derive image IDs
        raw_paths = [p.strip() for p in image_paths_raw.split(";") if p.strip()]

        # Resolve absolute paths relative to dataset_dir's parent
        # Paths in CSV are like: images/test/case_001/img_1.jpg
        abs_paths: List[str] = []
        for rp in raw_paths:
            candidate = self.dataset_dir / rp
            if candidate.exists():
                abs_paths.append(str(candidate))
            else:
                # Try as relative to dataset parent
                candidate2 = self.dataset_dir.parent / rp
                if candidate2.exists():
                    abs_paths.append(str(candidate2))
                else:
                    logger.warning("Image not found: %s", rp)
                    abs_paths.append(str(candidate))  # keep for error reporting

        image_ids = [Path(p).stem for p in raw_paths]  # e.g. img_1

        # Get user history
        user_history = self._history_map.get(user_id)
        if user_history is None:
            logger.warning("No history found for user %s — using defaults", user_id)
            user_history = UserHistory(user_id=user_id)

        # Get applicable evidence requirements
        applicable_reqs = self._get_applicable_requirements(claim_object)

        return ClaimContext(
            user_id=user_id,
            image_paths=image_paths_raw,
            user_claim=user_claim,
            claim_object=claim_object,
            image_path_list=abs_paths,
            image_ids=image_ids,
            user_history=user_history,
            applicable_requirements=applicable_reqs,
        )

    def _get_applicable_requirements(
        self, claim_object: str
    ) -> List[EvidenceRequirement]:
        """Return requirements matching this object type or 'all'."""
        return [
            r for r in self._requirements
            if r.claim_object in (claim_object, "all")
        ]

    def format_requirements_for_prompt(
        self, requirements: List[EvidenceRequirement]
    ) -> str:
        """Format evidence requirements as a readable string for prompts."""
        if not requirements:
            return "No specific evidence requirements found."
        lines = []
        for req in requirements:
            lines.append(
                f"- [{req.requirement_id}] ({req.applies_to}): {req.minimum_image_evidence}"
            )
        return "\n".join(lines)
