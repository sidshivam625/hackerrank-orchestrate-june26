
"""
pipeline/vlm_agent.py
──────────────────────
Layer 3 — Primary VLM reasoning using Gemini 2.5 Flash (Vertex AI).

Key design decisions:
- Uses Gemini's response_schema for schema-constrained JSON generation
  (token-level enforcement, not just prompt instructions)
- Supports context caching for the static system prompt (saves ~70% input tokens)
- Images are sent as base64-encoded inline data (no GCS upload needed for hackathon scale)
- Async execution with semaphore-based rate limiting
- Pydantic validation on every response with auto-retry on schema violation
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

import google.generativeai as genai
from google.generativeai import caching
import datetime

from models.schema import ClaimAnalysisResult, GEMINI_RESPONSE_SCHEMA
from pipeline.ingestion import ClaimContext, DataIngestionEngine
from utils.risk_scorer import compute_user_risk_flags, format_risk_context_for_prompt

logger = logging.getLogger(__name__)


def _load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    prompts_dir = Path(__file__).parent.parent / "prompts"
    path = prompts_dir / prompt_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _image_to_inline_data(image_path: str) -> Dict[str, Any]:
    """Convert an image file to Gemini inline data format."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return {"mime_type": mime_type, "data": data}


class GeminiVLMAgent:
    """
    Primary verification agent using Gemini 2.5 Flash.

    Features:
    - Schema-constrained JSON output via response_schema
    - Context caching of static system prompt
    - Async processing with semaphore rate limiting
    - Pydantic validation + retry on schema failure
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-preview-05-20",
        max_concurrent: int = 8,
        max_retries: int = 4,
        enable_cache: bool = True,
        rejection_ratio_threshold: float = 0.35,
        velocity_threshold: int = 3,
        self_consistency_samples: int = 1,
        sample_temperature: float = 0.4,
    ):
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.enable_cache = enable_cache
        self.rejection_ratio_threshold = rejection_ratio_threshold
        self.velocity_threshold = velocity_threshold
        # Self-consistency: draw K samples and majority-vote. K=1 disables it
        # (deterministic temperature-0 single pass).
        self.self_consistency_samples = max(1, int(self_consistency_samples))
        self.sample_temperature = sample_temperature

        self.use_vertex = False
        self.project_id = ""
        self.region = "us-central1"

        self._semaphore: Optional[asyncio.Semaphore] = None
        self._model: Optional[Any] = None
        self._cache: Optional[Any] = None
        self._system_prompt: str = ""
        self._claim_analysis_prompt_template: str = ""
        self._ingestion_engine: Optional[DataIngestionEngine] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self, api_key: Optional[str] = None) -> None:
        """Configure the Gemini client and load prompts."""
        # Load prompts
        self._system_prompt = _load_prompt("system_prompt.txt")
        self._claim_analysis_prompt_template = _load_prompt("claim_analysis_prompt.txt")

        # Determine if we should use Vertex AI (GCP) or Google AI Studio
        project_id = os.environ.get("GCP_PROJECT_ID", "")
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        
        # Prefer Vertex AI if explicitly configured, otherwise fall back to AI Studio if API key is present
        if project_id and project_id != "your-gcp-project-id":
            self.use_vertex = True
            self.project_id = project_id
            self.region = os.environ.get("GCP_REGION", "us-central1")
            
            import vertexai
            vertexai.init(project=self.project_id, location=self.region)
            logger.info("Configured Vertex AI GenerativeModel for Project=%s Region=%s", self.project_id, self.region)
        else:
            self.use_vertex = False
            key_to_use = api_key or gemini_api_key
            if key_to_use:
                genai.configure(api_key=key_to_use)
            else:
                genai.configure()
            logger.info("Configured Google AI Studio (google-generativeai) SDK")

        # Setup model (with or without context caching)
        if self.enable_cache:
            self._setup_with_cache()
        else:
            if self.use_vertex:
                from vertexai.generative_models import GenerativeModel as VertexGenerativeModel
                # Ensure clean model name on Vertex
                model_name = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
                self._model = VertexGenerativeModel(
                    model_name=model_name,
                    system_instruction=self._system_prompt,
                )
            else:
                self._model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=self._system_prompt,
                )

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        logger.info(
            "GeminiVLMAgent initialised — model=%s, cache=%s, concurrency=%d, platform=%s",
            self.model_name,
            self.enable_cache,
            self.max_concurrent,
            "VertexAI" if self.use_vertex else "AIStudio",
        )

    def _setup_with_cache(self) -> None:
        """
        Create a context cache for the static system prompt.
        This reduces input token costs by ~70% for the cached portion.
        Minimum 1,024 tokens required for caching.
        """
        model_name = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name

        if self.use_vertex:
            try:
                from vertexai.generative_models import GenerativeModel as VertexGenerativeModel
                from vertexai.preview import caching as vertex_caching

                ttl_secs = int(os.environ.get("CONTEXT_CACHE_TTL_SECONDS", "7200"))
                self._cache = vertex_caching.CachedContent.create(
                    model_name=model_name,
                    system_instruction=self._system_prompt,
                    ttl=datetime.timedelta(seconds=ttl_secs),
                )
                self._model = VertexGenerativeModel.from_cached_content(
                    cached_content=self._cache
                )
                logger.info("Vertex AI Context cache created: %s", self._cache.name)
            except Exception as e:
                logger.warning(
                    "Vertex AI Context cache creation failed (%s) — falling back to uncached model", e
                )
                from vertexai.generative_models import GenerativeModel as VertexGenerativeModel
                self._model = VertexGenerativeModel(
                    model_name=model_name,
                    system_instruction=self._system_prompt,
                )
                self._cache = None
        else:
            try:
                self._cache = caching.CachedContent.create(
                    model=self.model_name,
                    system_instruction=self._system_prompt,
                    ttl=datetime.timedelta(hours=2),
                )
                self._model = genai.GenerativeModel.from_cached_content(
                    cached_content=self._cache
                )
                logger.info("AI Studio Context cache created: %s", self._cache.name)
            except Exception as e:
                logger.warning(
                    "AI Studio Context cache creation failed (%s) — falling back to uncached model", e
                )
                self._model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=self._system_prompt,
                )
                self._cache = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyse_claim_async(
        self,
        ctx: ClaimContext,
        ingestion_engine: DataIngestionEngine,
    ) -> ClaimAnalysisResult:
        """
        Analyse a single claim asynchronously.
        Uses the semaphore to enforce concurrency limits.
        """
        async with self._semaphore:
            if self.self_consistency_samples <= 1:
                # Single pass. PRIMARY_TEMPERATURE defaults to 0.0 (deterministic).
                import os as _os
                t = float(_os.environ.get("PRIMARY_TEMPERATURE", "0.0"))
                return await self._analyse_with_retry(ctx, ingestion_engine, t)

            # Self-consistency: draw K diverse samples then majority-vote.
            results: List[ClaimAnalysisResult] = []
            for _ in range(self.self_consistency_samples):
                results.append(
                    await self._analyse_with_retry(
                        ctx, ingestion_engine, self.sample_temperature
                    )
                )
            voted = self._vote(results)
            logger.info(
                "Self-consistency for %s: %s -> %s",
                ctx.user_id,
                [r.claim_status for r in results],
                voted.claim_status,
            )
            return voted

    def analyse_claim_sync(
        self,
        ctx: ClaimContext,
        ingestion_engine: DataIngestionEngine,
    ) -> ClaimAnalysisResult:
        """Synchronous wrapper for single-claim analysis (used in evaluation)."""
        return asyncio.run(self.analyse_claim_async(ctx, ingestion_engine))

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    async def _analyse_with_retry(
        self,
        ctx: ClaimContext,
        ingestion_engine: DataIngestionEngine,
        temperature: float = 0.0,
    ) -> ClaimAnalysisResult:
        """Retry loop with exponential backoff + schema validation retry."""
        import random

        last_error: Optional[Exception] = None
        validation_error_msg: str = ""

        for attempt in range(self.max_retries):
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._call_gemini(ctx, ingestion_engine, validation_error_msg, temperature),
                )
                return result

            except Exception as e:
                last_error = e
                err_str = str(e)

                if "429" in err_str or "quota" in err_str.lower():
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "Rate limit hit for %s (attempt %d/%d) — waiting %.1fs",
                        ctx.user_id, attempt + 1, self.max_retries, wait
                    )
                    await asyncio.sleep(wait)
                elif "ValidationError" in err_str or "schema" in err_str.lower():
                    validation_error_msg = err_str[:300]
                    logger.warning(
                        "Schema validation failed for %s (attempt %d/%d): %s",
                        ctx.user_id, attempt + 1, self.max_retries, err_str[:200]
                    )
                    await asyncio.sleep(1)
                else:
                    logger.error(
                        "Unexpected error for %s: %s", ctx.user_id, err_str
                    )
                    await asyncio.sleep(2 ** attempt)

        # All retries exhausted — return a safe fallback
        logger.error(
            "All %d retries failed for %s: %s",
            self.max_retries, ctx.user_id, last_error
        )
        return self._build_fallback_result(ctx)

    def _call_gemini(
        self,
        ctx: ClaimContext,
        ingestion_engine: DataIngestionEngine,
        validation_error_msg: str = "",
        temperature: float = 0.0,
    ) -> ClaimAnalysisResult:
        """Build prompt, call Gemini, parse and validate response."""
        # Build the per-claim user-turn prompt
        user_prompt = self._build_user_prompt(ctx, ingestion_engine, validation_error_msg)

        if self.use_vertex:
            from vertexai.generative_models import Part, GenerationConfig as VertexGenerationConfig
            content_parts: List[Any] = [user_prompt]

            for img_path in ctx.image_path_list:
                try:
                    # Anchor each image to its image_id so the model can ground
                    # supporting_image_ids correctly on multi-image claims.
                    image_id = Path(img_path).stem
                    content_parts.append(f"=== IMAGE {image_id} ===")
                    inline = _image_to_inline_data(img_path)
                    raw_bytes = base64.b64decode(inline["data"])
                    part = Part.from_data(data=raw_bytes, mime_type=inline["mime_type"])
                    content_parts.append(part)
                except Exception as e:
                    logger.warning("Could not load image for Vertex AI %s: %s", img_path, e)
            
            # Pass as dict to avoid Vertex AI SDK GenerationConfig parsing bugs that truncate to 24 tokens
            generation_config = {
                "response_mime_type": "application/json",
                "response_schema": GEMINI_RESPONSE_SCHEMA,
                "temperature": temperature,   # 0.0 single-pass; >0 for self-consistency samples
                "max_output_tokens": 8192,
            }
        else:
            # Build content parts: [text_prompt, label1, image1, label2, image2, ...]
            content_parts = [{"text": user_prompt}]

            for img_path in ctx.image_path_list:
                try:
                    # Anchor each image to its image_id (see Vertex path above).
                    image_id = Path(img_path).stem
                    content_parts.append({"text": f"=== IMAGE {image_id} ==="})
                    inline = _image_to_inline_data(img_path)
                    content_parts.append({"inline_data": inline})
                except Exception as e:
                    logger.warning("Could not load image %s: %s", img_path, e)

            generation_config = genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=GEMINI_RESPONSE_SCHEMA,
                temperature=temperature,   # 0.0 single-pass; >0 for self-consistency samples
                max_output_tokens=4096,
            )

        response = self._model.generate_content(
            content_parts,
            generation_config=generation_config,
        )

        raw_text = response.text.strip()
        logger.debug("Gemini raw response for %s: %s", ctx.user_id, raw_text[:200])

        # Parse JSON
        data = self._parse_json_response(raw_text)

        # Validate with Pydantic (normalises enum values)
        result = ClaimAnalysisResult(**data)

        # Merge pre-computed risk flags with model-generated flags
        result = self._merge_risk_flags(result, ctx)

        return result

    def _build_user_prompt(
        self,
        ctx: ClaimContext,
        ingestion_engine: DataIngestionEngine,
        validation_error_msg: str = "",
    ) -> str:
        """Build the per-claim analysis prompt with all runtime context."""
        # Compute user risk flags
        risk_flags = compute_user_risk_flags(
            ctx.user_history,
            self.rejection_ratio_threshold,
            self.velocity_threshold,
        )
        risk_context = format_risk_context_for_prompt(ctx.user_history, risk_flags)

        # Format evidence requirements
        req_text = ingestion_engine.format_requirements_for_prompt(
            ctx.applicable_requirements
        )

        # Format image quality flags
        quality_flags = ctx.image_quality_flags
        quality_str = (
            ";".join(quality_flags) if quality_flags else "none"
        )

        # Input hardening: cap the conversation length fed to the model so an
        # adversarially huge user_claim can't blow the token budget / cost.
        # Only the prompt copy is truncated — the OUTPUT user_claim is untouched.
        MAX_CLAIM_CHARS = 6000
        safe_user_claim = ctx.user_claim
        if len(safe_user_claim) > MAX_CLAIM_CHARS:
            safe_user_claim = safe_user_claim[:MAX_CLAIM_CHARS] + " …[truncated]"

        # Load template and substitute
        template = self._claim_analysis_prompt_template
        prompt = template.format(
            user_id=ctx.user_id,
            claim_object=ctx.claim_object,
            user_claim=safe_user_claim,
            image_quality_flags=quality_str,
            user_history_flags=";".join(
                [f for f in risk_flags] if risk_flags else ["none"]
            ),
            user_history_summary=ctx.user_history.history_summary if ctx.user_history else "Unknown",
            evidence_requirement=req_text[:500],
            evidence_requirement_detail=req_text,
        )

        # Append validation error if retrying
        if validation_error_msg:
            prompt += (
                f"\n\nPREVIOUS ATTEMPT FAILED SCHEMA VALIDATION:\n{validation_error_msg}\n"
                "Please ensure your JSON exactly matches the required schema."
            )

        return prompt

    def _merge_risk_flags(
        self,
        result: ClaimAnalysisResult,
        ctx: ClaimContext,
    ) -> ClaimAnalysisResult:
        """
        Merge pre-computed risk flags (from OpenCV + user history)
        into the model's output risk_flags.
        """
        model_flags = set(
            f.strip() for f in result.risk_flags.split(";") if f.strip() and f.strip() != "none"
        )
        pre_flags = set(ctx.image_quality_flags + ctx.computed_risk_flags)

        # Forcefully remove only the flags that are OWNED by a deterministic
        # layer (OpenCV quality pre-pass + user-history rules). These are
        # injected via `pre_flags` below, so letting the model also emit them
        # would create duplicate / conflicting signals.
        #
        # NOTE: possible_manipulation and non_original_image are intentionally
        # NOT in this set — they are *content* authenticity judgments that only
        # the VLM can make from the pixels (screenshot, photo-of-photo, spliced
        # image). The rule layer never produces them, so stripping them here
        # made it impossible for the system to ever emit them, even though the
        # ground truth and the test set (e.g. "screenshots instead of original
        # photos") require them. Final injection/watermark detection happens
        # deterministically in postprocessor.py via transcribed image text.
        forbidden_model_flags = {
            "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
            "user_history_risk", "manual_review_required",
        }
        model_flags = model_flags - forbidden_model_flags

        merged = model_flags | pre_flags
        if not merged:
            merged = {"none"}

        result.risk_flags = ";".join(sorted(merged)) if merged != {"none"} else "none"
        return result

    @staticmethod
    def _parse_json_response(raw_text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response, handling markdown code blocks."""
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("```").strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract first JSON object
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Cannot parse JSON from response: {clean[:200]}")

    @staticmethod
    def _build_fallback_result(ctx: ClaimContext) -> ClaimAnalysisResult:
        """Safe fallback when all retries fail."""
        return ClaimAnalysisResult(
            evidence_standard_met=False,
            evidence_standard_met_reason="System error — could not complete analysis",
            risk_flags="manual_review_required",
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification="Automated analysis failed — manual review required",
            supporting_image_ids="none",
            valid_image=False,
            severity="unknown",
        )

    @staticmethod
    def _vote(results: List[ClaimAnalysisResult]) -> ClaimAnalysisResult:
        """
        Majority-vote K self-consistency samples into one result.

        - claim_status: plurality vote (ties broken by the more conservative
          outcome: not_enough_information < contradicted < supported is the
          severity of asserting damage, so on a tie prefer abstaining).
        - issue_type / object_part / severity / valid_image: modal value among
          the samples that agree with the winning claim_status.
        - risk_flags: a flag is kept only if it appears in a MAJORITY of
          samples. Deterministic pre-flags (OpenCV / history) appear in every
          sample so they always survive; inconsistent content flags (e.g. a
          one-off possible_manipulation) are filtered out — exactly the noise
          we want gone.
        """
        from collections import Counter

        if len(results) == 1:
            return results[0]

        k = len(results)
        status_counts = Counter(r.claim_status for r in results)
        top = status_counts.most_common()
        best = top[0][1]
        tied = [s for s, c in top if c == best]
        if len(tied) == 1:
            winner = tied[0]
        else:
            # Tie-break: prefer the more conservative verdict.
            order = {"not_enough_information": 0, "contradicted": 1, "supported": 2}
            winner = min(tied, key=lambda s: order.get(s, 1))

        agree = [r for r in results if r.claim_status == winner] or results

        def modal(attr: str) -> Any:
            return Counter(getattr(r, attr) for r in agree).most_common(1)[0][0]

        base = agree[0]
        base.claim_status = winner
        base.issue_type = modal("issue_type")
        base.object_part = modal("object_part")
        base.severity = modal("severity")
        base.valid_image = modal("valid_image")
        base.evidence_standard_met = modal("evidence_standard_met")
        base.claim_status_justification = modal("claim_status_justification")
        base.supporting_image_ids = modal("supporting_image_ids")

        # Keep the fullest image-text transcription across samples (so injection /
        # non-original detection isn't dropped by a sample that missed the text).
        texts = [getattr(r, "detected_image_text", "") or "" for r in results]
        non_empty = [t for t in texts if t.strip() and t.strip().lower() != "none"]
        if non_empty:
            base.detected_image_text = max(non_empty, key=len)

        threshold = (k // 2) + 1
        flag_counts: Dict[str, int] = {}
        for r in results:
            for f in r.risk_flags.split(";"):
                f = f.strip()
                if f and f != "none":
                    flag_counts[f] = flag_counts.get(f, 0) + 1
        kept = sorted(f for f, c in flag_counts.items() if c >= threshold)
        base.risk_flags = ";".join(kept) if kept else "none"
        return base

    def cleanup(self) -> None:
        """Delete context cache to avoid storage charges."""
        if self._cache:
            try:
                self._cache.delete()
                logger.info("Context cache deleted")
            except Exception as e:
                logger.warning("Could not delete cache: %s", e)
