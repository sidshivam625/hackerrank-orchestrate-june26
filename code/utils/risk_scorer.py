"""
utils/risk_scorer.py
──────────────────────
Rule-based user history risk scoring.

Applies deterministic logic BEFORE any LLM call.
Pre-computed risk flags are injected into the VLM prompt as context,
grounding the model's risk assessment in objective rules.
"""

from __future__ import annotations

import logging
from typing import List

from pipeline.ingestion import UserHistory

logger = logging.getLogger(__name__)

# Risk thresholds (configurable via env in main.py)
DEFAULT_REJECTION_RATIO_THRESHOLD = 0.35
DEFAULT_VELOCITY_THRESHOLD = 3


def compute_user_risk_flags(
    history: UserHistory,
    rejection_ratio_threshold: float = DEFAULT_REJECTION_RATIO_THRESHOLD,
    velocity_threshold: int = DEFAULT_VELOCITY_THRESHOLD,
) -> List[str]:
    """
    Evaluate user history and return a list of risk flags.

    Rules applied:
    1. Rejection ratio > threshold AND past_claim_count >= 5 → user_history_risk
    2. last_90_days_claim_count > velocity_threshold → manual_review_required
    3. history_flags contains known risk keywords → pass through directly
    4. history_summary contains fraud/suspicious keywords → user_history_risk

    These flags do NOT override visual evidence — they are context-only.
    """
    flags: List[str] = []

    if history is None:
        return flags

    # ---- Rule 1: Historical rejection ratio ----
    if (
        history.rejection_ratio > rejection_ratio_threshold
        and history.past_claim_count >= 5
    ):
        if "user_history_risk" not in flags:
            flags.append("user_history_risk")
        logger.debug(
            "User %s flagged: rejection_ratio=%.2f (threshold=%.2f)",
            history.user_id,
            history.rejection_ratio,
            rejection_ratio_threshold,
        )

    # ---- Rule 2: Claim velocity (last 90 days) ----
    if history.last_90_days_claim_count > velocity_threshold:
        if "manual_review_required" not in flags:
            flags.append("manual_review_required")
        logger.debug(
            "User %s flagged: last_90_days_claim_count=%d (threshold=%d)",
            history.user_id,
            history.last_90_days_claim_count,
            velocity_threshold,
        )

    # ---- Rule 3: Parse existing history_flags field ----
    KNOWN_RISK_FLAGS = {
        "user_history_risk", "manual_review_required",
        "possible_manipulation", "non_original_image",
    }
    if history.history_flags and history.history_flags.lower() not in ("none", ""):
        for raw_flag in history.history_flags.split(";"):
            flag = raw_flag.strip().lower()
            if flag in KNOWN_RISK_FLAGS and flag not in flags:
                flags.append(flag)

    # ---- Rule 4: Semantic keyword parsing in history_summary ----
    import re
    HIGH_RISK_KEYWORDS = [
        r"\bfraud", r"mismatch", r"manipulat", r"suspicious",
        r"fabricat", r"\breject\b", r"exaggerat", r"similar image",
        r"screenshot", r"non-original"
    ]
    summary_lower = history.history_summary.lower()
    for keyword in HIGH_RISK_KEYWORDS:
        if re.search(keyword, summary_lower):
            if "user_history_risk" not in flags:
                flags.append("user_history_risk")
            break

    return flags


def format_risk_context_for_prompt(
    history: UserHistory,
    risk_flags: List[str],
) -> str:
    """
    Format user history and pre-computed risk flags as a readable
    string for injection into the VLM prompt.
    """
    flag_str = ";".join(risk_flags) if risk_flags else "none"

    lines = [
        f"User History Summary: {history.history_summary or 'No summary available'}",
        f"Past Claims: {history.past_claim_count} total "
        f"({history.accept_claim} accepted, "
        f"{history.rejected_claim} rejected, "
        f"{history.manual_review_claim} manual review)",
        f"Claims in Last 90 Days: {history.last_90_days_claim_count}",
        f"Rejection Ratio: {history.rejection_ratio:.0%}",
        f"Pre-computed Risk Flags: {flag_str}",
    ]

    if risk_flags:
        lines.append(
            "NOTE: These risk flags add context only. "
            "If visual evidence clearly supports the claim, "
            "claim_status should still be 'supported' — "
            "but include the risk flags in your risk_flags output."
        )

    return "\n".join(lines)
