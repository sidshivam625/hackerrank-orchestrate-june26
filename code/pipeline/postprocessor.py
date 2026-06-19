"""
pipeline/postprocessor.py
──────────────────────────
Layer 5 — Post-processing, schema compliance, and deterministic overrides.

After the VLM returns its analysis, this layer:
1. Applies deterministic evidence_standard_met logic (rule-based, not LLM)
2. Validates and normalises all enum fields
3. Assembles the final OutputRow
4. Enforces column order and value constraints
"""

from __future__ import annotations

import logging
from typing import List, Optional

from models.schema import ClaimAnalysisResult, OutputRow
from pipeline.ingestion import ClaimContext, EvidenceRequirement

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed values for schema compliance
# ---------------------------------------------------------------------------

VALID_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}
VALID_ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown"
}
VALID_SEVERITY = {"none", "low", "medium", "high", "unknown"}
VALID_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed",
    "low_light_or_glare", "wrong_angle", "wrong_object",
    "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk",
    "manual_review_required"
}

CAR_PARTS = {
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender",
    "quarter_panel", "body", "unknown"
}
LAPTOP_PARTS = {
    "screen", "keyboard", "trackpad", "hinge", "lid",
    "corner", "port", "base", "body", "unknown"
}
PACKAGE_PARTS = {
    "box", "package_corner", "package_side", "seal",
    "label", "contents", "item", "unknown"
}

OBJECT_PART_MAP = {
    "car": CAR_PARTS,
    "laptop": LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}

ONTOLOGY_MAP = {
    "car": {
        "windshield": {"crack", "glass_shatter", "scratch", "none", "unknown"},
        "headlight": {"crack", "glass_shatter", "scratch", "broken_part", "none", "unknown"},
        "taillight": {"crack", "glass_shatter", "scratch", "broken_part", "none", "unknown"},
        "side_mirror": {"crack", "glass_shatter", "scratch", "broken_part", "missing_part", "none", "unknown"},
        "door": {"dent", "scratch", "crack", "broken_part", "none", "unknown"},
        "hood": {"dent", "scratch", "crack", "broken_part", "none", "unknown"},
        "front_bumper": {"dent", "scratch", "crack", "broken_part", "missing_part", "none", "unknown"},
        "rear_bumper": {"dent", "scratch", "crack", "broken_part", "missing_part", "none", "unknown"},
        "fender": {"dent", "scratch", "crack", "broken_part", "none", "unknown"},
        "quarter_panel": {"dent", "scratch", "crack", "broken_part", "none", "unknown"},
        "body": {"dent", "scratch", "crack", "broken_part", "none", "unknown"},
        "unknown": VALID_ISSUE_TYPES
    },
    "laptop": {
        "screen": {"crack", "glass_shatter", "stain", "water_damage", "none", "unknown"},
        "keyboard": {"broken_part", "missing_part", "water_damage", "stain", "none", "unknown"},
        "trackpad": {"broken_part", "scratch", "water_damage", "none", "unknown"},
        "hinge": {"broken_part", "crack", "none", "unknown"},
        "lid": {"dent", "scratch", "crack", "none", "unknown"},
        "corner": {"dent", "scratch", "crack", "none", "unknown"},
        "port": {"broken_part", "water_damage", "none", "unknown"},
        "base": {"dent", "scratch", "crack", "none", "unknown"},
        "body": {"dent", "scratch", "crack", "water_damage", "stain", "none", "unknown"},
        "unknown": VALID_ISSUE_TYPES
    },
    "package": {
        "box": {"torn_packaging", "crushed_packaging", "water_damage", "stain", "dent", "none", "unknown"},
        "package_corner": {"crushed_packaging", "torn_packaging", "dent", "water_damage", "stain", "none", "unknown"},
        "package_side": {"crushed_packaging", "torn_packaging", "water_damage", "stain", "dent", "none", "unknown"},
        "seal": {"torn_packaging", "none", "unknown"},
        "label": {"water_damage", "stain", "torn_packaging", "none", "unknown"},
        "contents": {"missing_part", "broken_part", "glass_shatter", "water_damage", "none", "unknown"},
        "item": {"missing_part", "broken_part", "glass_shatter", "water_damage", "none", "unknown"},
        "unknown": VALID_ISSUE_TYPES
    }
}


class PostProcessor:
    """
    Applies deterministic post-processing to VLM results.
    This is the final compliance layer before writing output.csv.
    """

    def assemble_output_row(
        self,
        ctx: ClaimContext,
        result: ClaimAnalysisResult,
    ) -> OutputRow:
        """
        Combine ClaimContext + VLM result into a fully validated OutputRow.
        Applies deterministic overrides and schema compliance fixes.
        """
        # 1. Deterministic evidence_standard_met check
        evidence_met, evidence_reason = self._check_evidence_standard(ctx, result)

        # 2. Validate and normalise object_part for the claim_object
        object_part = self._normalise_object_part(result.object_part, ctx.claim_object)

        # 3. Normalise risk_flags (validate each, filter invalid)
        risk_flags = self._normalise_risk_flags(result.risk_flags)

        # 4. Validate issue_type against ontology
        issue_type = result.issue_type if result.issue_type in VALID_ISSUE_TYPES else "unknown"
        allowed_issues = ONTOLOGY_MAP.get(ctx.claim_object, {}).get(object_part, VALID_ISSUE_TYPES)
        if issue_type not in allowed_issues:
            logger.warning(
                "Ontology mismatch for %s: %s is not allowed for %s on %s. Setting to unknown.",
                ctx.user_id, issue_type, object_part, ctx.claim_object
            )
            issue_type = "unknown"

        # 5. Validate claim_status
        claim_status = (
            result.claim_status
            if result.claim_status in VALID_CLAIM_STATUS
            else "not_enough_information"
        )

        # 5.1 Contradiction resolution
        if claim_status == "supported" and "wrong_object" in risk_flags:
            claim_status = "contradicted"

        # 6. Validate and override severity
        severity = result.severity if result.severity in VALID_SEVERITY else "unknown"
        if claim_status == "contradicted" and issue_type == "none":
            severity = "none"
        elif claim_status == "not_enough_information":
            severity = "unknown"
        elif claim_status == "supported" and severity == "unknown":
            severity = "medium"
        elif claim_status == "supported" and severity == "none":
            severity = "low"

        # 7. If evidence standard not met AND no valid images at all, override
        # Only override when there are truly zero usable images (hard block).
        # Do NOT override when the VLM has valid images to work with.
        if not evidence_met and claim_status == "supported" and ctx.valid_images_count == 0:
            claim_status = "not_enough_information"
            logger.debug(
                "Override: claim_status -> not_enough_information for %s "
                "(no valid images at all)",
                ctx.user_id
            )

        # 8. valid_image: Use VLM's assessment if images exist, else use OpenCV
        valid_image = result.valid_image if ctx.valid_images_count > 0 else False

        # 9. Normalise supporting_image_ids
        supporting_ids = self._normalise_image_ids(
            result.supporting_image_ids, ctx.image_ids
        )

        return OutputRow(
            user_id=ctx.user_id,
            image_paths=ctx.image_paths,
            user_claim=ctx.user_claim,
            claim_object=ctx.claim_object,
            evidence_standard_met=evidence_met,
            evidence_standard_met_reason=evidence_reason or result.evidence_standard_met_reason,
            risk_flags=risk_flags,
            issue_type=issue_type,
            object_part=object_part,
            claim_status=claim_status,
            claim_status_justification=result.claim_status_justification,
            supporting_image_ids=supporting_ids,
            valid_image=valid_image,
            severity=severity,
        )

    # ------------------------------------------------------------------
    # Deterministic evidence standard check
    # ------------------------------------------------------------------

    def _check_evidence_standard(
        self,
        ctx: ClaimContext,
        result: ClaimAnalysisResult,
    ) -> tuple[bool, Optional[str]]:
        """
        Deterministic evidence_standard_met logic.
        Only override VLM when there are zero usable images.
        Otherwise trust the VLM's assessment.

        Returns (evidence_met: bool, reason: Optional[str])
        """
        # No valid images at all → evidence NOT met
        if ctx.valid_images_count == 0:
            return (
                False,
                "No usable images were provided — all images failed quality checks.",
            )

        # If at least one image passed OpenCV checks, trust the VLM's assessment.
        # The VLM has seen the actual images and is better positioned to judge
        # evidence sufficiency than rule-based logic with aggregated flags.
        return result.evidence_standard_met, None

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_object_part(raw_part: str, claim_object: str) -> str:
        """Ensure object_part is valid for the given claim_object."""
        part = raw_part.strip().lower()
        allowed = OBJECT_PART_MAP.get(claim_object, set())
        return part if part in allowed else "unknown"

    @staticmethod
    def _normalise_risk_flags(raw_flags: str) -> str:
        """Filter risk_flags to only allowed values."""
        if not raw_flags or raw_flags.strip().lower() == "none":
            return "none"

        flags = [f.strip().lower() for f in raw_flags.split(";") if f.strip()]
        valid = [f for f in flags if f in VALID_RISK_FLAGS]
        return ";".join(valid) if valid else "none"

    @staticmethod
    def _normalise_image_ids(raw_ids: str, available_ids: List[str]) -> str:
        """
        Validate supporting_image_ids against the actual image IDs in the claim.
        Filters to only IDs that exist in the claim's image set.
        """
        if not raw_ids or raw_ids.strip().lower() == "none":
            return "none"

        ids = [i.strip() for i in raw_ids.split(";") if i.strip()]
        # Validate each ID exists in the actual image set
        valid = [i for i in ids if i in available_ids]
        return ";".join(valid) if valid else "none"
