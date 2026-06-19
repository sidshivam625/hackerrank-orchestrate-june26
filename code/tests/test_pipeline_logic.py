#!/usr/bin/env python3
"""
tests/test_pipeline_logic.py
────────────────────────────
Deterministic unit tests for the pipeline's decision logic.

These tests make NO API calls — they exercise the pure functions that decide
risk-flag merging, ensemble voting, and post-processing heuristics. Run them
to confirm that the recent correctness fixes behave as intended and to guard
against regressions.

Usage:
    python tests/test_pipeline_logic.py     # plain runner, exits non-zero on failure
    pytest tests/test_pipeline_logic.py     # if pytest is installed
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the code/ root importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.schema import ClaimAnalysisResult
from pipeline.ingestion import ClaimContext, UserHistory
from pipeline.escalation_agent import ensemble_vote, _merge_flags
from pipeline.postprocessor import PostProcessor
from pipeline.vlm_agent import GeminiVLMAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(**overrides) -> ClaimAnalysisResult:
    """Build a ClaimAnalysisResult with sensible defaults."""
    base = dict(
        evidence_standard_met=True,
        evidence_standard_met_reason="reason",
        risk_flags="none",
        issue_type="dent",
        object_part="rear_bumper",
        claim_status="supported",
        claim_status_justification="img_1 shows the damage",
        supporting_image_ids="img_1",
        valid_image=True,
        severity="medium",
    )
    base.update(overrides)
    return ClaimAnalysisResult(**base)


def _ctx(**overrides) -> ClaimContext:
    base = dict(
        user_id="user_test",
        image_paths="images/test/case_x/img_1.jpg",
        user_claim="claim",
        claim_object="car",
        image_path_list=["img_1.jpg"],
        image_ids=["img_1"],
        user_history=UserHistory(user_id="user_test"),
        valid_images_count=1,
    )
    base.update(overrides)
    return ClaimContext(**base)


def _flags(s: str) -> set:
    return set(f for f in s.split(";") if f and f != "none")


# ---------------------------------------------------------------------------
# A — VLM risk-flag merge keeps authenticity flags, strips owned flags
# ---------------------------------------------------------------------------

def test_merge_keeps_authenticity_flags():
    """possible_manipulation / non_original_image must survive the merge."""
    agent = GeminiVLMAgent()  # no initialise() — we only call a pure method
    result = _result(
        claim_status="contradicted",
        risk_flags="claim_mismatch;possible_manipulation;non_original_image",
    )
    ctx = _ctx(image_quality_flags=[], computed_risk_flags=[])
    merged = agent._merge_risk_flags(result, ctx)
    flags = _flags(merged.risk_flags)
    assert "possible_manipulation" in flags, flags
    assert "non_original_image" in flags, flags
    assert "claim_mismatch" in flags, flags


def test_merge_strips_layer_owned_flags_from_model():
    """Model-emitted OpenCV/history flags are dropped (the layers own them)."""
    agent = GeminiVLMAgent()
    result = _result(risk_flags="blurry_image;user_history_risk;wrong_angle")
    ctx = _ctx(image_quality_flags=[], computed_risk_flags=[])
    merged = agent._merge_risk_flags(result, ctx)
    flags = _flags(merged.risk_flags)
    assert "blurry_image" not in flags, flags
    assert "user_history_risk" not in flags, flags
    # content flag the model is allowed to own is retained
    assert "wrong_angle" in flags, flags


def test_merge_injects_precomputed_flags():
    """OpenCV + history pre-flags are unioned into the output."""
    agent = GeminiVLMAgent()
    result = _result(risk_flags="none")
    ctx = _ctx(
        image_quality_flags=["blurry_image"],
        computed_risk_flags=["user_history_risk", "manual_review_required"],
    )
    merged = agent._merge_risk_flags(result, ctx)
    flags = _flags(merged.risk_flags)
    assert {"blurry_image", "user_history_risk", "manual_review_required"} <= flags, flags


# ---------------------------------------------------------------------------
# D — ensemble voting has no directional ("supported") bias
# ---------------------------------------------------------------------------

def test_ensemble_agreement_merges_flags():
    primary = _result(claim_status="supported", risk_flags="user_history_risk")
    secondary = _result(claim_status="supported", risk_flags="wrong_angle")
    out = ensemble_vote(primary, secondary)
    assert out.claim_status == "supported"
    assert {"user_history_risk", "wrong_angle"} <= _flags(out.risk_flags)


def test_ensemble_does_not_flip_contradicted_to_supported():
    """The key regression: a correct 'contradicted' must NOT be overridden
    just because the second model leans 'supported'."""
    primary = _result(claim_status="contradicted", issue_type="scratch")
    secondary = _result(claim_status="supported")
    out = ensemble_vote(primary, secondary)
    assert out.claim_status == "contradicted", out.claim_status
    assert "manual_review_required" in _flags(out.risk_flags)


def test_ensemble_resolves_nei_with_decisive_secondary():
    """When the primary abstains, a decisive second opinion resolves it."""
    primary = _result(claim_status="not_enough_information", issue_type="unknown")
    secondary = _result(claim_status="contradicted", issue_type="none")
    out = ensemble_vote(primary, secondary)
    assert out.claim_status == "contradicted", out.claim_status
    assert "manual_review_required" in _flags(out.risk_flags)


def test_ensemble_keeps_decisive_primary_over_nei_secondary():
    primary = _result(claim_status="supported")
    secondary = _result(claim_status="not_enough_information")
    out = ensemble_vote(primary, secondary)
    assert out.claim_status == "supported", out.claim_status


def test_ensemble_none_secondary_passthrough():
    primary = _result(claim_status="supported")
    assert ensemble_vote(primary, None) is primary


# ---------------------------------------------------------------------------
# B — post-processor adds manual_review_required to contradicted claims
# ---------------------------------------------------------------------------

def test_postprocessor_contradicted_gets_manual_review():
    pp = PostProcessor()
    ctx = _ctx(valid_images_count=1)
    result = _result(
        claim_status="contradicted",
        issue_type="scratch",
        object_part="rear_bumper",
        severity="low",
        risk_flags="claim_mismatch",
    )
    row = pp.assemble_output_row(ctx, result)
    assert row.claim_status == "contradicted"
    assert "manual_review_required" in _flags(row.risk_flags), row.risk_flags


def test_postprocessor_supported_strips_contradiction_flags():
    pp = PostProcessor()
    ctx = _ctx(valid_images_count=1)
    result = _result(
        claim_status="supported",
        risk_flags="claim_mismatch;wrong_object",  # inconsistent with 'supported'
    )
    row = pp.assemble_output_row(ctx, result)
    flags = _flags(row.risk_flags)
    assert "claim_mismatch" not in flags, flags
    assert "wrong_object" not in flags, flags


def test_postprocessor_ontology_gating():
    """An issue_type impossible for the (object, part) pair becomes 'unknown'."""
    pp = PostProcessor()
    ctx = _ctx(claim_object="car", valid_images_count=1)
    # torn_packaging is not valid for a car windshield
    result = _result(
        claim_status="supported",
        issue_type="torn_packaging",
        object_part="windshield",
    )
    row = pp.assemble_output_row(ctx, result)
    assert row.issue_type == "unknown", row.issue_type


def test_postprocessor_invalid_object_part_for_object():
    pp = PostProcessor()
    ctx = _ctx(claim_object="laptop", valid_images_count=1)
    result = _result(object_part="front_bumper")  # car part on a laptop
    row = pp.assemble_output_row(ctx, result)
    assert row.object_part == "unknown", row.object_part


# ---------------------------------------------------------------------------
# Self-consistency vote (_vote)
# ---------------------------------------------------------------------------

def test_vote_majority_claim_status():
    rs = [_result(claim_status="supported"),
          _result(claim_status="supported"),
          _result(claim_status="contradicted")]
    out = GeminiVLMAgent._vote(rs)
    assert out.claim_status == "supported", out.claim_status


def test_vote_tie_breaks_conservative():
    # 2-way tie supported vs contradicted -> prefer the more conservative
    rs = [_result(claim_status="supported"),
          _result(claim_status="contradicted")]
    out = GeminiVLMAgent._vote(rs)
    assert out.claim_status == "contradicted", out.claim_status


def test_vote_flag_needs_majority():
    # possible_manipulation appears in only 1/3 -> dropped; user-content flag in 2/3 -> kept
    rs = [_result(claim_status="contradicted", risk_flags="claim_mismatch;possible_manipulation"),
          _result(claim_status="contradicted", risk_flags="claim_mismatch"),
          _result(claim_status="contradicted", risk_flags="claim_mismatch")]
    out = GeminiVLMAgent._vote(rs)
    flags = _flags(out.risk_flags)
    assert "claim_mismatch" in flags, flags
    assert "possible_manipulation" not in flags, flags


def test_vote_single_sample_passthrough():
    r = _result(claim_status="supported")
    assert GeminiVLMAgent._vote([r]) is r


def test_vote_modal_fields_from_winners():
    rs = [_result(claim_status="supported", issue_type="dent", severity="medium"),
          _result(claim_status="supported", issue_type="dent", severity="low"),
          _result(claim_status="contradicted", issue_type="scratch", severity="none")]
    out = GeminiVLMAgent._vote(rs)
    assert out.issue_type == "dent", out.issue_type  # modal among supported winners


# ---------------------------------------------------------------------------
# _merge_flags utility
# ---------------------------------------------------------------------------

def test_merge_flags_dedupes_and_sorts():
    out = _merge_flags("b;a", "a;c", extra=["d"])
    assert out == "a;b;c;d", out


def test_merge_flags_all_none():
    assert _merge_flags("none", "none") == "none"


# ---------------------------------------------------------------------------
# Plain runner (no pytest dependency)
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # pragma: no cover
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
