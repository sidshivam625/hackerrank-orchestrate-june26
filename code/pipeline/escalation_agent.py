"""
pipeline/escalation_agent.py
──────────────────────────────
Layer 4 — Qwen2.5-VL-72B escalation via AIML API.

Triggered when the primary Gemini agent returns:
- claim_status = "not_enough_information" AND images are valid
- Contradictory signals between risk_flags and claim_status
- Multiple images with conflicting evidence

Uses the OpenAI-compatible AIML API endpoint.
Results are ensemble-voted with the primary result.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from models.schema import ClaimAnalysisResult
from pipeline.ingestion import ClaimContext

logger = logging.getLogger(__name__)


def _load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    prompts_dir = Path(__file__).parent.parent / "prompts"
    path = prompts_dir / prompt_name
    return path.read_text(encoding="utf-8")


def _should_escalate(
    primary_result: ClaimAnalysisResult,
    ctx: ClaimContext,
    enable_escalation: bool = True,
) -> bool:
    """
    Decide if a claim should be escalated to Qwen for second opinion.

    Escalation criteria (any one triggers it):
    1. claim_status = "not_enough_information" AND at least one valid image
    2. evidence_standard_met=False but images appear physically valid
    3. Multiple conflicting risk flags suggesting ambiguity
    """
    if not enable_escalation:
        return False

    # Don't escalate if no valid images (Qwen can't help)
    if ctx.valid_images_count == 0:
        return False

    # Escalate uncertain claims with valid images
    if (
        primary_result.claim_status == "not_enough_information"
        and primary_result.valid_image
    ):
        return True

    # Escalate if evidence standard not met but images are valid
    if not primary_result.evidence_standard_met and primary_result.valid_image:
        return True

    # Escalate on contradicted claims from high-risk users (cross-check)
    if (
        primary_result.claim_status == "contradicted"
        and ctx.user_history
        and ctx.user_history.rejection_ratio > 0.5
    ):
        return True

    return False


class QwenEscalationAgent:
    """
    Secondary verification agent using Qwen3-VL-32B via AIML API.
    OpenAI-compatible interface.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.aimlapi.com/v1",
        model_name: str = "alibaba/qwen3-vl-32b-instruct",
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self._client: Optional[AsyncOpenAI] = None
        self._system_prompt: str = ""
        self._escalation_prompt_template: str = ""
        self._api_key = api_key
        self._base_url = base_url

    def initialise(self) -> None:
        """Set up the AIML API client and load prompts."""
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
        self._system_prompt = _load_prompt("system_prompt.txt")
        self._escalation_prompt_template = _load_prompt("escalation_prompt.txt")
        logger.info(
            "QwenEscalationAgent initialised — model=%s", self.model_name
        )

    async def analyse_claim_async(
        self,
        ctx: ClaimContext,
        primary_result: ClaimAnalysisResult,
    ) -> Optional[ClaimAnalysisResult]:
        """
        Send claim to Qwen for independent second-opinion analysis.
        Returns None if escalation fails or is not applicable.
        """
        import random

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                result = await self._call_qwen(ctx, primary_result)
                logger.info(
                    "Escalation complete for %s — Qwen result: %s",
                    ctx.user_id, result.claim_status
                )
                return result
            except Exception as e:
                last_error = e
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Escalation attempt %d/%d failed for %s: %s — waiting %.1fs",
                    attempt + 1, self.max_retries, ctx.user_id, str(e)[:100], wait
                )
                await asyncio.sleep(wait)

        logger.error(
            "All escalation retries failed for %s: %s",
            ctx.user_id, last_error
        )
        return None

    async def _call_qwen(
        self,
        ctx: ClaimContext,
        primary_result: ClaimAnalysisResult,
    ) -> ClaimAnalysisResult:
        """Build multimodal message and call Qwen-VL model."""
        # Build prompt from template
        prompt = self._escalation_prompt_template.format(
            user_claim=ctx.user_claim,
            claim_object=ctx.claim_object,
            claim_context=f"Object: {ctx.claim_object}, User says: {ctx.user_claim[:200]}",
            primary_result=primary_result.claim_status,
            primary_flags=primary_result.risk_flags,
            evidence_requirement="\n".join(
                f"- {r.applies_to}: {r.minimum_image_evidence}"
                for r in ctx.applicable_requirements
            ),
        )

        # Build message content with images
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for img_path in ctx.image_path_list:
            try:
                # Normalize and resize for Qwen via PIL.
                # (1) Handles CMYK / progressive / palette JPEGs that Qwen rejects.
                # (2) Resizes to MAX_DIM on the longest side — AIML API enforces a
                #     per-image file-size limit; high-res phone photos can exceed it.
                import io
                from PIL import Image as _PILImage
                MAX_DIM = int(os.environ.get("QWEN_MAX_IMAGE_DIM", "1568"))
                JPEG_QUALITY = int(os.environ.get("QWEN_JPEG_QUALITY", "85"))
                with _PILImage.open(img_path) as pil_img:
                    if pil_img.mode not in ("RGB",):
                        pil_img = pil_img.convert("RGB")
                    w, h = pil_img.size
                    if max(w, h) > MAX_DIM:
                        scale = MAX_DIM / max(w, h)
                        pil_img = pil_img.resize(
                            (int(w * scale), int(h * scale)),
                            _PILImage.LANCZOS,
                        )
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=JPEG_QUALITY)
                img_data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
                mime = "image/jpeg"

                # Anchor each image to its image_id so supporting_image_ids is
                # grounded correctly on multi-image claims.
                image_id = Path(img_path).stem
                content.append({"type": "text", "text": f"=== IMAGE {image_id} ==="})
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{img_data}",
                        "detail": "high"
                    }
                })
            except Exception as e:
                logger.warning("Could not encode image %s for Qwen: %s", img_path, e)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": content},
        ]

        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8192,
            temperature=0.0,   # Deterministic decoding for reproducibility
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content.strip()
        logger.debug("Qwen raw response for %s: %s", ctx.user_id, raw_text[:200])

        data = self._parse_json(raw_text)

        # Remove escalation-specific fields before Pydantic validation
        data.pop("escalation_confidence", None)
        data.pop("escalation_notes", None)
        data.pop("image_analysis", None)

        return ClaimAnalysisResult(**data)

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """Parse JSON from Qwen response."""
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        if not clean.startswith("{"):
            clean = "{" + clean
        if not clean.endswith("}"):
            clean = clean + "}"
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Cannot parse Qwen JSON: {clean[:200]}")


def ensemble_vote(
    primary: ClaimAnalysisResult,
    secondary: Optional[ClaimAnalysisResult],
) -> ClaimAnalysisResult:
    """
    Combine primary (Gemini) and secondary (Qwen) results.

    Voting logic (no directional bias):
    - If both agree on claim_status → keep that result, merge flags.
    - If they disagree AND the primary was uncertain
      (`not_enough_information`) → adopt the secondary's decisive verdict,
      because resolving that uncertainty is the whole reason we escalated.
      Flag the row for human review.
    - If they disagree AND the primary was already decisive
      (`supported`/`contradicted`) → KEEP the primary verdict and flag for
      review. We deliberately do NOT prefer "supported" over "contradicted":
      that old tie-break inflated false positives on the contradicted class
      (our weakest class) by overriding correct contradictions whenever the
      second model leaned supportive.
    - risk_flags from both models are always merged.
    """
    if secondary is None:
        return primary

    # Both agree
    if primary.claim_status == secondary.claim_status:
        logger.info(
            "Ensemble: both models agree — %s", primary.claim_status
        )
        primary.risk_flags = _merge_flags(primary.risk_flags, secondary.risk_flags)
        return primary

    # Disagreement: only let the second opinion override when the primary
    # abstained (not_enough_information) and the secondary is decisive.
    decisive = {"supported", "contradicted"}
    if (
        primary.claim_status == "not_enough_information"
        and secondary.claim_status in decisive
    ):
        logger.info(
            "Ensemble: Qwen resolves NEI → %s (Gemini was not_enough_information)",
            secondary.claim_status,
        )
        secondary.risk_flags = _merge_flags(
            primary.risk_flags, secondary.risk_flags, extra=["manual_review_required"]
        )
        # Preserve the primary's image-text transcription (Qwen doesn't produce it),
        # so deterministic injection / non-original detection survives escalation.
        if not getattr(secondary, "detected_image_text", "") and getattr(primary, "detected_image_text", ""):
            secondary.detected_image_text = primary.detected_image_text
        return secondary

    # Primary was decisive (or secondary is NEI): keep the primary verdict and
    # surface the disagreement for human review.
    logger.info(
        "Ensemble: keeping Gemini (%s) over Qwen (%s) — flagged for review",
        primary.claim_status, secondary.claim_status,
    )
    primary.risk_flags = _merge_flags(
        primary.risk_flags, secondary.risk_flags, extra=["manual_review_required"]
    )
    return primary


def _merge_flags(*flag_strings: str, extra: Optional[List[str]] = None) -> str:
    """Merge multiple semicolon-separated flag strings."""
    all_flags: set = set()
    for fs in flag_strings:
        for f in fs.split(";"):
            f = f.strip()
            if f and f != "none":
                all_flags.add(f)
    if extra:
        all_flags.update(extra)

    return ";".join(sorted(all_flags)) if all_flags else "none"
