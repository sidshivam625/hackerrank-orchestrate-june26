#!/usr/bin/env python3
"""
code/main.py
────────────
Multi-Modal Claims Verification System — Main Entry Point

Reads dataset/claims.csv and produces output.csv with structured
damage claim verification results.

Usage:
    python main.py [--dataset-dir PATH] [--output PATH] [--sample]
    python main.py --help

Environment variables (set in .env file):
    See .env.example for full list.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Setup: load .env and configure logging BEFORE importing pipeline modules
# ---------------------------------------------------------------------------

# Load .env from code/ directory or repository root
dotenv_code = Path(__file__).parent / ".env"
dotenv_root = Path(__file__).parent.parent / ".env"
if dotenv_code.exists():
    load_dotenv(dotenv_code)
elif dotenv_root.exists():
    load_dotenv(dotenv_root)
else:
    load_dotenv()

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Pipeline imports (after env is loaded)
# ---------------------------------------------------------------------------

import pandas as pd

from pipeline.ingestion import DataIngestionEngine, ClaimContext
from pipeline.image_validator import ImageValidator
from pipeline.vlm_agent import GeminiVLMAgent
from pipeline.escalation_agent import QwenEscalationAgent, _should_escalate, ensemble_vote
from pipeline.postprocessor import PostProcessor
from models.schema import OutputRow
from utils.risk_scorer import compute_user_risk_flags


# ---------------------------------------------------------------------------
# Output columns (must be in this exact order per problem_statement.md)
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part",
    "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity"
]


# ---------------------------------------------------------------------------
# Core processing orchestrator
# ---------------------------------------------------------------------------

class ClaimsVerificationOrchestrator:
    """
    Coordinates all 5 pipeline layers to process a batch of claims.

    Layer 1: DataIngestionEngine — data loading & joining
    Layer 2: ImageValidator — OpenCV quality pre-pass
    Layer 3: GeminiVLMAgent — primary VLM reasoning
    Layer 4: QwenEscalationAgent — second-opinion on uncertain claims
    Layer 5: PostProcessor — schema compliance & output assembly
    """

    def __init__(self):
        # Layer 1
        self.ingestion = DataIngestionEngine(
            dataset_dir=os.environ.get("DATASET_DIR", "../dataset")
        )
        # Layer 2
        self.validator = ImageValidator(
            blur_threshold=float(os.environ.get("BLUR_THRESHOLD", "80.0")),
            brightness_min=float(os.environ.get("BRIGHTNESS_MIN", "50.0")),
            brightness_max=float(os.environ.get("BRIGHTNESS_MAX", "210.0")),
            entropy_threshold=float(os.environ.get("ENTROPY_THRESHOLD", "3.0")),
            min_width=int(os.environ.get("MIN_IMAGE_WIDTH", "200")),
            min_height=int(os.environ.get("MIN_IMAGE_HEIGHT", "200")),
        )
        # Layer 3
        self.gemini_agent = GeminiVLMAgent(
            model_name=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20"),
            max_concurrent=int(os.environ.get("MAX_CONCURRENT_REQUESTS", "8")),
            max_retries=int(os.environ.get("MAX_RETRIES", "4")),
            enable_cache=os.environ.get("ENABLE_CONTEXT_CACHE", "true").lower() == "true",
            rejection_ratio_threshold=float(os.environ.get("USER_REJECTION_RATIO_THRESHOLD", "0.35")),
            velocity_threshold=int(os.environ.get("USER_VELOCITY_THRESHOLD", "3")),
        )
        # Layer 4
        self.qwen_agent: Optional[QwenEscalationAgent] = None
        aiml_key = os.environ.get("AIML_API_KEY", "")
        self.enable_escalation = (
            os.environ.get("ENABLE_ESCALATION", "true").lower() == "true"
            and bool(aiml_key)
            and aiml_key != "your-aiml-api-key"
        )
        if self.enable_escalation:
            self.qwen_agent = QwenEscalationAgent(
                api_key=aiml_key,
                base_url=os.environ.get("AIML_API_BASE_URL", "https://api.aimlapi.com/v1"),
                model_name=os.environ.get("ESCALATION_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct"),
            )
        # Layer 5
        self.postprocessor = PostProcessor()

        # Metrics tracking
        self.stats = {
            "total": 0,
            "escalated": 0,
            "errors": 0,
            "supported": 0,
            "contradicted": 0,
            "not_enough_information": 0,
            "start_time": 0.0,
        }

    def initialise(self) -> None:
        """Initialise all agents."""
        logger.info("Initialising pipeline agents…")
        self.gemini_agent.initialise()
        if self.qwen_agent:
            self.qwen_agent.initialise()
        logger.info(
            "Pipeline ready — Gemini=%s, Escalation=%s",
            os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            "enabled" if self.enable_escalation else "disabled"
        )

    async def process_all_async(
        self, contexts: List[ClaimContext]
    ) -> List[OutputRow]:
        """Process all claims concurrently."""
        tasks = [self._process_one(ctx) for ctx in contexts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output_rows: List[OutputRow] = []
        for ctx, result in zip(contexts, results):
            if isinstance(result, Exception):
                logger.error("Failed to process %s: %s", ctx.user_id, result)
                output_rows.append(self._build_error_row(ctx))
                self.stats["errors"] += 1
            else:
                output_rows.append(result)
                self.stats[result.claim_status] = self.stats.get(result.claim_status, 0) + 1

        return output_rows

    async def _process_one(self, ctx: ClaimContext) -> OutputRow:
        """Full 5-layer pipeline for a single claim."""
        # ---- Layer 2: Image validation ----
        overall_valid, quality_flags, img_results = self.validator.validate_image_set(
            ctx.image_path_list
        )
        ctx.image_quality_flags = quality_flags
        ctx.valid_images_count = sum(1 for r in img_results if r.valid)

        # ---- Pre-compute user risk flags (Rule-based) ----
        user_risk_flags = compute_user_risk_flags(ctx.user_history)
        ctx.computed_risk_flags = user_risk_flags

        # ---- Layer 3: Primary VLM (Gemini) ----
        primary_result = await self.gemini_agent.analyse_claim_async(ctx, self.ingestion)

        # ---- Layer 4: Escalation (Qwen) if needed ----
        final_result = primary_result
        if self.qwen_agent and _should_escalate(primary_result, ctx, self.enable_escalation):
            logger.info("Escalating %s to Qwen…", ctx.user_id)
            self.stats["escalated"] += 1
            secondary = await self.qwen_agent.analyse_claim_async(ctx, primary_result)
            final_result = ensemble_vote(primary_result, secondary)

        # ---- Layer 5: Post-processing ----
        output_row = self.postprocessor.assemble_output_row(ctx, final_result)
        self.stats["total"] += 1

        logger.info(
            "✓ %s — %s | %s | flags: %s",
            ctx.user_id,
            output_row.claim_status,
            output_row.issue_type,
            output_row.risk_flags
        )

        return output_row

    @staticmethod
    def _build_error_row(ctx: ClaimContext) -> OutputRow:
        """Fallback row for processing failures."""
        return OutputRow(
            user_id=ctx.user_id,
            image_paths=ctx.image_paths,
            user_claim=ctx.user_claim,
            claim_object=ctx.claim_object,
            evidence_standard_met=False,
            evidence_standard_met_reason="Processing error — manual review required",
            risk_flags="manual_review_required",
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification="System error — could not complete automated analysis",
            supporting_image_ids="none",
            valid_image=False,
            severity="unknown",
        )

    def print_summary(self) -> None:
        """Print processing summary statistics."""
        elapsed = time.time() - self.stats["start_time"]
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("  Total claims processed : %d", self.stats["total"])
        logger.info("  Supported              : %d", self.stats.get("supported", 0))
        logger.info("  Contradicted           : %d", self.stats.get("contradicted", 0))
        logger.info("  Not enough information : %d", self.stats.get("not_enough_information", 0))
        logger.info("  Escalated to Qwen      : %d", self.stats.get("escalated", 0))
        logger.info("  Errors                 : %d", self.stats.get("errors", 0))
        logger.info("  Elapsed time           : %.1fs", elapsed)
        logger.info("=" * 60)

    def cleanup(self) -> None:
        """Clean up resources (e.g., delete context cache)."""
        self.gemini_agent.cleanup()


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_output_csv(rows: List[OutputRow], output_path: str) -> None:
    """Write output rows to CSV in the correct column order."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "user_id": row.user_id,
                "image_paths": row.image_paths,
                "user_claim": row.user_claim,
                "claim_object": row.claim_object,
                "evidence_standard_met": str(row.evidence_standard_met).lower(),
                "evidence_standard_met_reason": row.evidence_standard_met_reason,
                "risk_flags": row.risk_flags,
                "issue_type": row.issue_type,
                "object_part": row.object_part,
                "claim_status": row.claim_status,
                "claim_status_justification": row.claim_status_justification,
                "supporting_image_ids": row.supporting_image_ids,
                "valid_image": str(row.valid_image).lower(),
                "severity": row.severity,
            })

    logger.info("Output written to %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Modal Claims Verification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process test claims (claims.csv → output.csv)
  python main.py

  # Custom paths
  python main.py --dataset-dir /path/to/dataset --output /path/to/output.csv

  # Run on sample data (for evaluation/testing)
  python main.py --sample
        """
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("DATASET_DIR", "../dataset"),
        help="Path to the dataset directory (default: ../dataset)"
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("OUTPUT_CSV", "../output.csv"),
        help="Output CSV path (default: ../output.csv)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process sample_claims.csv instead of claims.csv"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of claims to process (for testing)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    """Main async entry point."""
    orchestrator = ClaimsVerificationOrchestrator()
    orchestrator.stats["start_time"] = time.time()

    try:
        # Load data
        orchestrator.ingestion.dataset_dir = Path(args.dataset_dir)
        if args.sample:
            orchestrator.ingestion.load_sample()
            logger.info("Processing sample_claims.csv")
        else:
            orchestrator.ingestion.load_all()
            logger.info("Processing claims.csv")

        contexts = orchestrator.ingestion.get_claim_contexts()

        if args.limit:
            contexts = contexts[:args.limit]
            logger.info("Limited to %d claims", args.limit)

        logger.info("Loaded %d claims to process", len(contexts))

        # Initialise agents
        orchestrator.initialise()

        # Process all claims
        output_rows = await orchestrator.process_all_async(contexts)

        # Write output
        write_output_csv(output_rows, args.output)

        # Print summary
        orchestrator.print_summary()

    finally:
        orchestrator.cleanup()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
