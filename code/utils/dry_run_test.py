#!/usr/bin/env python3
"""
utils/dry_run_test.py
──────────────────────
Smoke-test the pipeline WITHOUT making any API calls.

Tests:
1. Data ingestion and joining
2. OpenCV image quality validation on sample images
3. Risk scorer rule evaluation
4. Post-processor schema compliance
5. Output CSV structure validation

Usage:
    python utils/dry_run_test.py
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("dry_run")


def test_ingestion():
    """Test Layer 1: Data loading and joining."""
    from pipeline.ingestion import DataIngestionEngine

    # Resolve dataset dir relative to this script's location
    script_dir = Path(__file__).parent
    default_dataset = str(script_dir.parent.parent / "dataset")
    dataset_dir = os.environ.get("DATASET_DIR", default_dataset)
    engine = DataIngestionEngine(dataset_dir)
    engine.load_sample()
    contexts = engine.get_claim_contexts()

    assert len(contexts) > 0, "No contexts loaded!"
    ctx = contexts[0]
    assert ctx.user_id, "user_id is empty"
    assert ctx.image_path_list, "No image paths parsed"
    assert ctx.user_history is not None, "User history not joined"
    assert len(ctx.applicable_requirements) > 0, "No evidence requirements matched"

    logger.info("✓ Ingestion: %d claims loaded, %d evidence reqs for first claim",
                len(contexts), len(ctx.applicable_requirements))
    return contexts


def test_image_validator(contexts):
    """Test Layer 2: OpenCV quality checks."""
    from pipeline.image_validator import ImageValidator

    validator = ImageValidator()
    tested = 0
    for ctx in contexts[:5]:  # Test first 5 claims
        if not ctx.image_path_list:
            continue
        valid, flags, results = validator.validate_image_set(ctx.image_path_list)
        assert isinstance(valid, bool), "valid must be bool"
        assert isinstance(flags, list), "flags must be list"
        tested += 1
        logger.info("  %s: images=%d valid=%s flags=%s",
                    ctx.user_id, len(results), valid, flags or "none")

    assert tested > 0, "No images were tested!"
    logger.info("✓ ImageValidator: tested %d claims", tested)


def test_risk_scorer(contexts):
    """Test rule-based user risk scoring."""
    from utils.risk_scorer import compute_user_risk_flags

    for ctx in contexts[:10]:
        flags = compute_user_risk_flags(ctx.user_history)
        assert isinstance(flags, list), "Risk flags must be a list"

    # Test high-risk user
    from pipeline.ingestion import UserHistory
    risky = UserHistory(
        user_id="test_risky",
        past_claim_count=10,
        rejected_claim=5,
        last_90_days_claim_count=5,
        history_flags="user_history_risk",
        history_summary="Several exaggerated claims"
    )
    flags = compute_user_risk_flags(risky)
    assert "user_history_risk" in flags, f"Expected user_history_risk in {flags}"
    assert "manual_review_required" in flags, f"Expected manual_review_required in {flags}"

    # Test low-risk user
    safe = UserHistory(
        user_id="test_safe",
        past_claim_count=2,
        rejected_claim=0,
        last_90_days_claim_count=1,
        history_flags="none",
        history_summary="Low-risk user"
    )
    flags = compute_user_risk_flags(safe)
    assert flags == [], f"Expected no flags for safe user, got {flags}"

    logger.info("✓ RiskScorer: correctly flags high-risk and clear low-risk users")


def test_postprocessor(contexts):
    """Test Layer 5: Schema compliance."""
    from pipeline.postprocessor import PostProcessor
    from models.schema import ClaimAnalysisResult

    pp = PostProcessor()
    ctx = contexts[0]
    ctx.valid_images_count = 1

    # Mock VLM result
    mock_result = ClaimAnalysisResult(
        evidence_standard_met=True,
        evidence_standard_met_reason="Test",
        risk_flags="none",
        issue_type="dent",
        object_part="rear_bumper",
        claim_status="supported",
        claim_status_justification="Test justification referencing img_1",
        supporting_image_ids="img_1",
        valid_image=True,
        severity="medium",
    )

    row = pp.assemble_output_row(ctx, mock_result)
    assert row.claim_status == "supported"
    assert row.issue_type == "dent"
    assert row.severity == "medium"
    assert row.user_id == ctx.user_id

    # Test invalid object_part remapped to unknown
    mock_result.object_part = "invalid_part_xyz"
    mock_result.claim_status = "contradicted"
    row2 = pp.assemble_output_row(ctx, mock_result)
    assert row2.object_part == "unknown", f"Expected 'unknown' for invalid part, got {row2.object_part}"

    logger.info("✓ PostProcessor: schema compliance and override logic work correctly")


def test_output_schema():
    """Test Pydantic schema validation."""
    from models.schema import ClaimAnalysisResult

    # Valid result
    r = ClaimAnalysisResult(
        evidence_standard_met=True,
        evidence_standard_met_reason="OK",
        risk_flags=["blurry_image", "user_history_risk"],  # list form
        issue_type="scratch",
        object_part="front_bumper",
        claim_status="supported",
        claim_status_justification="img_1 shows scratch",
        supporting_image_ids="img_1",
        valid_image=True,
        severity="low",
    )
    assert r.risk_flags == "blurry_image;user_history_risk", f"Got: {r.risk_flags}"

    # Invalid risk flag filtered out
    r2 = ClaimAnalysisResult(
        evidence_standard_met=False,
        evidence_standard_met_reason="Bad",
        risk_flags="blurry_image;INVALID_FLAG;low_light_or_glare",
        issue_type="unknown",
        object_part="unknown",
        claim_status="not_enough_information",
        claim_status_justification="Cannot evaluate",
        supporting_image_ids="none",
        valid_image=False,
        severity="unknown",
    )
    assert "INVALID_FLAG" not in r2.risk_flags, "Invalid flag should be filtered"
    assert "blurry_image" in r2.risk_flags

    logger.info("✓ Pydantic schema: enum validation and list normalisation work")


def main():
    logger.info("=" * 50)
    logger.info("DRY RUN TEST — No API calls made")
    logger.info("=" * 50)

    try:
        contexts = test_ingestion()
        test_image_validator(contexts)
        test_risk_scorer(contexts)
        test_postprocessor(contexts)
        test_output_schema()

        logger.info("=" * 50)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("The pipeline structure is valid.")
        logger.info("Next step: configure .env and run: python main.py --limit 3")
        logger.info("=" * 50)

    except AssertionError as e:
        logger.error("TEST FAILED: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("UNEXPECTED ERROR: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
