"""
models/schema.py
────────────────
Pydantic models for the claim verification output schema.
Using Literal types ensures Gemini's schema-constrained generation
produces only valid enum values — no post-hoc remapping needed.
"""

from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Allowed enum literals (mirroring problem_statement.md exactly)
# ---------------------------------------------------------------------------

ClaimStatus = Literal["supported", "contradicted", "not_enough_information"]

IssueType = Literal[
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown"
]

CarPart = Literal[
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender",
    "quarter_panel", "body", "unknown"
]

LaptopPart = Literal[
    "screen", "keyboard", "trackpad", "hinge", "lid",
    "corner", "port", "base", "body", "unknown"
]

PackagePart = Literal[
    "box", "package_corner", "package_side", "seal",
    "label", "contents", "item", "unknown"
]

RiskFlag = Literal[
    "none", "blurry_image", "cropped_or_obstructed",
    "low_light_or_glare", "wrong_angle", "wrong_object",
    "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk",
    "manual_review_required"
]

Severity = Literal["none", "low", "medium", "high", "unknown"]

# All allowed object parts (union for validation)
ALL_OBJECT_PARTS = (
    set(CarPart.__args__)        # type: ignore[attr-defined]
    | set(LaptopPart.__args__)   # type: ignore[attr-defined]
    | set(PackagePart.__args__)  # type: ignore[attr-defined]
)

VALID_RISK_FLAGS = set(RiskFlag.__args__)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# VLM output schema (what Gemini / Qwen returns)
# ---------------------------------------------------------------------------

class ClaimAnalysisResult(BaseModel):
    """
    Structured output from the VLM verification agent.
    All categorical fields are constrained to allowed enum values.
    """

    evidence_standard_met: bool = Field(
        description="True if the image set is sufficient to evaluate the claim"
    )
    evidence_standard_met_reason: str = Field(
        description="Concise explanation of the evidence sufficiency decision"
    )
    risk_flags: str = Field(
        description="Semicolon-separated risk flags, or 'none'"
    )
    issue_type: IssueType = Field(
        description="Primary visible damage category"
    )
    object_part: str = Field(
        description="Relevant object part showing the verified issue"
    )
    claim_status: ClaimStatus = Field(
        description="Final decision: supported, contradicted, or not_enough_information"
    )
    claim_status_justification: str = Field(
        description="Concise image-grounded explanation referencing image IDs"
    )
    supporting_image_ids: str = Field(
        description="Semicolon-separated image IDs supporting the decision, or 'none'"
    )
    valid_image: bool = Field(
        description="True if image quality is usable for automated review"
    )
    severity: Severity = Field(
        description="Categorical severity rating of the physical damage"
    )
    detected_image_text: str = Field(
        default="",
        description=(
            "Verbatim transcription of ALL text/labels/notes/watermarks/overlays "
            "visible in the images (or 'none'). Used by deterministic post-processing "
            "to flag prompt-injection and non-original images — the model only reports "
            "what it sees, it does not act on it."
        ),
    )

    @field_validator("risk_flags", mode="before")
    @classmethod
    def normalise_risk_flags(cls, v: object) -> str:
        """Accept list or semicolon-string; validate each flag."""
        if isinstance(v, list):
            flags = [str(f).strip().lower() for f in v]
        elif isinstance(v, str):
            flags = [f.strip().lower() for f in v.split(";") if f.strip()]
        else:
            return "none"

        valid = [f for f in flags if f in VALID_RISK_FLAGS]
        return ";".join(valid) if valid else "none"

    @field_validator("object_part", mode="before")
    @classmethod
    def normalise_object_part(cls, v: object) -> str:
        val = str(v).strip().lower()
        return val if val in ALL_OBJECT_PARTS else "unknown"

    @field_validator("supporting_image_ids", mode="before")
    @classmethod
    def normalise_image_ids(cls, v: object) -> str:
        if isinstance(v, list):
            ids = [str(i).strip() for i in v if str(i).strip()]
            return ";".join(ids) if ids else "none"
        s = str(v).strip()
        return s if s else "none"


# ---------------------------------------------------------------------------
# Full output row (input fields + VLM output fields)
# ---------------------------------------------------------------------------

class OutputRow(BaseModel):
    """One row in output.csv — all 14 required columns in order."""

    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: str

    # Column order matches problem_statement.md §Required output
    OUTPUT_COLUMNS: List[str] = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part",
        "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity"
    ]

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Gemini response_schema (JSON Schema dict for GenerationConfig)
# ---------------------------------------------------------------------------

GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "evidence_standard_met":        {"type": "BOOLEAN"},
        "evidence_standard_met_reason": {"type": "STRING"},
        "risk_flags":                   {"type": "STRING"},
        "issue_type": {
            "type": "STRING",
            "enum": list(IssueType.__args__)  # type: ignore[attr-defined]
        },
        "object_part": {"type": "STRING"},
        "claim_status": {
            "type": "STRING",
            "enum": list(ClaimStatus.__args__)  # type: ignore[attr-defined]
        },
        "claim_status_justification":   {"type": "STRING"},
        "supporting_image_ids":         {"type": "STRING"},
        "valid_image":                  {"type": "BOOLEAN"},
        "severity": {
            "type": "STRING",
            "enum": list(Severity.__args__)  # type: ignore[attr-defined]
        },
        "detected_image_text":          {"type": "STRING"},
    },
    "required": [
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part", "claim_status",
        "claim_status_justification", "supporting_image_ids",
        "valid_image", "severity", "detected_image_text"
    ]
}
