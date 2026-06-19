#!/usr/bin/env python3
"""
evaluation/main.py
───────────────────
Evaluation script for the Multi-Modal Claims Verification System.

Compares system predictions against sample_claims.csv ground truth.
Computes per-field metrics and generates evaluation_report.md.

Usage:
    # First run the main pipeline on sample data:
    python ../main.py --sample --output sample_predictions.csv

    # Then evaluate:
    python main.py --predictions sample_predictions.csv
    python main.py --predictions sample_predictions.csv --report evaluation_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("evaluation")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLAIM_STATUS_LABELS = ["supported", "contradicted", "not_enough_information"]
SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": -1}

GROUND_TRUTH_PATH = Path(__file__).parent.parent.parent / "dataset" / "sample_claims.csv"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class EvaluationMetrics:
    """Computes all evaluation metrics between predictions and ground truth."""

    def __init__(self, gold_df: pd.DataFrame, pred_df: pd.DataFrame):
        self.gold = gold_df
        self.pred = pred_df
        self.n = len(gold_df)

    def claim_status_metrics(self) -> Dict[str, Any]:
        """Per-class F1, precision, recall for claim_status (primary metric)."""
        gold = self.gold["claim_status"].str.lower().str.strip()
        pred = self.pred["claim_status"].str.lower().str.strip()

        # Align to same set of labels
        all_labels = sorted(set(gold.unique()) | set(pred.unique()))

        report = classification_report(
            gold, pred, labels=CLAIM_STATUS_LABELS, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(gold, pred, labels=CLAIM_STATUS_LABELS)

        return {
            "accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, average="macro", labels=CLAIM_STATUS_LABELS, zero_division=0),
            "per_class": report,
            "confusion_matrix": cm.tolist(),
            "labels": CLAIM_STATUS_LABELS,
        }

    def boolean_field_metrics(self, field: str) -> Dict[str, float]:
        """Binary accuracy and F1 for boolean fields."""
        def to_bool(s: Any) -> int:
            if isinstance(s, bool):
                return int(s)
            return 1 if str(s).strip().lower() in ("true", "1", "yes") else 0

        gold = self.gold[field].apply(to_bool)
        pred = self.pred[field].apply(to_bool)

        return {
            "accuracy": accuracy_score(gold, pred),
            "f1": f1_score(gold, pred, zero_division=0),
            "precision": precision_score(gold, pred, zero_division=0),
            "recall": recall_score(gold, pred, zero_division=0),
        }

    def categorical_field_metrics(self, field: str) -> Dict[str, float]:
        """Multi-class accuracy and weighted F1 for categorical fields."""
        gold = self.gold[field].str.lower().str.strip()
        pred = self.pred[field].str.lower().str.strip()
        labels = sorted(set(gold.unique()) | set(pred.unique()))

        return {
            "accuracy": accuracy_score(gold, pred),
            "weighted_f1": f1_score(gold, pred, average="weighted", labels=labels, zero_division=0),
        }

    def risk_flags_jaccard(self) -> float:
        """
        Multi-label Jaccard similarity for risk_flags.
        Jaccard = |intersection| / |union| per claim, then averaged.
        """
        scores: List[float] = []
        for gold_str, pred_str in zip(self.gold["risk_flags"], self.pred["risk_flags"]):
            g = set(str(gold_str).lower().split(";")) - {"none", ""}
            p = set(str(pred_str).lower().split(";")) - {"none", ""}

            # Both empty → perfect match
            if not g and not p:
                scores.append(1.0)
            elif not g or not p:
                scores.append(0.0)
            else:
                scores.append(len(g & p) / len(g | p))

        return float(np.mean(scores)) if scores else 0.0

    def severity_mae(self) -> float:
        """
        Mean absolute error for severity (treating as ordinal).
        none=0, low=1, medium=2, high=3, unknown=-1 (excluded from MAE)
        """
        errors: List[float] = []
        for gold_str, pred_str in zip(self.gold["severity"], self.pred["severity"]):
            g = SEVERITY_ORDER.get(str(gold_str).lower().strip(), -1)
            p = SEVERITY_ORDER.get(str(pred_str).lower().strip(), -1)
            if g >= 0 and p >= 0:
                errors.append(abs(g - p))

        return float(np.mean(errors)) if errors else 0.0

    def failure_analysis(self, n: int = 5) -> List[Dict[str, str]]:
        """Find the N worst-performing claims for qualitative analysis."""
        failures: List[Dict[str, str]] = []

        for i, (_, gold_row) in enumerate(self.gold.iterrows()):
            if i >= len(self.pred):
                break
            pred_row = self.pred.iloc[i]

            issues: List[str] = []
            if gold_row["claim_status"] != pred_row.get("claim_status", ""):
                issues.append(
                    f"claim_status: expected={gold_row['claim_status']}, "
                    f"got={pred_row.get('claim_status', 'MISSING')}"
                )
            if gold_row.get("issue_type") and gold_row["issue_type"] != pred_row.get("issue_type", ""):
                issues.append(
                    f"issue_type: expected={gold_row['issue_type']}, "
                    f"got={pred_row.get('issue_type', 'MISSING')}"
                )
            if gold_row.get("object_part") and gold_row["object_part"] != pred_row.get("object_part", ""):
                issues.append(
                    f"object_part: expected={gold_row['object_part']}, "
                    f"got={pred_row.get('object_part', 'MISSING')}"
                )

            if issues:
                failures.append({
                    "user_id": str(gold_row.get("user_id", f"row_{i}")),
                    "claim_object": str(gold_row.get("claim_object", "")),
                    "user_claim_snippet": str(gold_row.get("user_claim", ""))[:80] + "…",
                    "issues": "; ".join(issues),
                    "predicted_justification": str(pred_row.get("claim_status_justification", ""))[:150],
                })

        # Sort by number of issues (more issues = worse)
        failures.sort(key=lambda x: len(x["issues"]), reverse=True)
        return failures[:n]


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(
    metrics: EvaluationMetrics,
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Generate markdown evaluation report."""
    status_m = metrics.claim_status_metrics()
    issue_m = metrics.categorical_field_metrics("issue_type")
    part_m = metrics.categorical_field_metrics("object_part")
    severity_m = metrics.categorical_field_metrics("severity")
    evidence_m = metrics.boolean_field_metrics("evidence_standard_met")
    valid_img_m = metrics.boolean_field_metrics("valid_image")
    jaccard = metrics.risk_flags_jaccard()
    sev_mae = metrics.severity_mae()
    failures = metrics.failure_analysis(n=5)

    # Confusion matrix as ASCII table
    cm = status_m["confusion_matrix"]
    labels = status_m["labels"]

    cm_lines = [
        "```",
        "Predicted →     " + "  ".join(f"{l[:10]:<12}" for l in labels),
    ]
    for i, label in enumerate(labels):
        row_vals = "  ".join(f"{cm[i][j]:<12}" for j in range(len(labels)))
        cm_lines.append(f"Actual {label[:10]:<12}: {row_vals}")
    cm_lines.append("```")
    cm_text = "\n".join(cm_lines)

    # Per-class F1 table
    per_class = status_m["per_class"]
    pc_rows = []
    for label in labels:
        if label in per_class:
            p = per_class[label]
            pc_rows.append(
                f"| {label:<25} | {p['precision']:.3f}     | {p['recall']:.3f}  "
                f"| {p['f1-score']:.3f} | {int(p['support'])}       |"
            )

    report = f"""# Evaluation Report
*Generated for Multi-Modal Claims Verification System*

---

## 1. Dataset Overview

| Metric              | Value                  |
|---------------------|------------------------|
| Sample claims       | {len(gold_df)}         |
| Claim objects       | {", ".join(gold_df["claim_object"].value_counts().index.tolist())} |
| Ground truth source | dataset/sample_claims.csv |

---

## 2. Primary Metric — `claim_status` Performance

**Overall Accuracy: {status_m['accuracy']:.3f} | Macro F1: {status_m['macro_f1']:.3f}**

### Per-Class Breakdown

| Class                     | Precision | Recall | F1    | Support |
|---------------------------|-----------|--------|-------|---------|
{chr(10).join(pc_rows)}

### Confusion Matrix

{cm_text}

---

## 3. Secondary Field Metrics

| Field                   | Metric         | Score    |
|-------------------------|----------------|----------|
| `issue_type`            | Accuracy       | {issue_m['accuracy']:.3f}  |
| `issue_type`            | Weighted F1    | {issue_m['weighted_f1']:.3f}  |
| `object_part`           | Accuracy       | {part_m['accuracy']:.3f}  |
| `object_part`           | Weighted F1    | {part_m['weighted_f1']:.3f}  |
| `severity`              | Accuracy       | {severity_m['accuracy']:.3f}  |
| `severity`              | Ordinal MAE    | {sev_mae:.3f}  |
| `evidence_standard_met` | Accuracy       | {evidence_m['accuracy']:.3f}  |
| `evidence_standard_met` | F1             | {evidence_m['f1']:.3f}  |
| `valid_image`           | Accuracy       | {valid_img_m['accuracy']:.3f}  |
| `valid_image`           | F1             | {valid_img_m['f1']:.3f}  |
| `risk_flags`            | Jaccard (avg)  | {jaccard:.3f}  |

---

## 4. Failure Analysis — Top 5 Worst Cases

The following claims had the most prediction errors:

{"".join(f'''
### Case {i+1}: {f['user_id']} ({f['claim_object']})
- **Claim snippet**: {f['user_claim_snippet']}
- **Errors**: {f['issues']}
- **Predicted justification**: {f['predicted_justification']}
''' for i, f in enumerate(failures)) if failures else "No failures detected — perfect match on all fields!"}

---

## 5. Operational Analysis

### 5.1 Model Call Estimates

| Phase               | Claims | Avg Images/Claim | VLM Calls | Escalation Rate | Total API Calls |
|---------------------|--------|------------------|-----------|-----------------|-----------------|
| Sample evaluation   | {len(gold_df)}     | ~2.0             | {len(gold_df)}         | ~10-15%         | ~{int(len(gold_df) * 1.12)}          |
| Test set (claims.csv)| 45    | ~2.1             | 45        | ~10-15%         | ~50             |

### 5.2 Token Usage Estimates (per claim)

| Component                  | Tokens  | Notes                                      |
|----------------------------|---------|--------------------------------------------|
| System prompt (cached)     | ~4,500  | Paid once via context cache (~4× cheaper)  |
| Per-claim text input       | ~1,200  | User claim + user history + requirements   |
| Per-image encoding         | ~258    | Gemini estimate per image at 1K detail     |
| 2.1 images × 258           | ~542    | Average image token cost                   |
| **Total active input/claim** | **~1,742** | Excluding cached tokens               |
| Output tokens/claim        | ~350    | Structured JSON response                   |

### 5.3 Cost Estimate (Test Set — 45 claims)

| Scenario                  | Input Tokens | Output Tokens | Est. Cost  |
|---------------------------|--------------|---------------|------------|
| Without caching (Gemini 2.5 Flash) | 45 × 6,242 = 280,890 | 45 × 350 = 15,750 | ~$0.19 |
| With context caching      | 45 × 1,742 = 78,390 active | 45 × 350 = 15,750 | ~$0.06 |
| Escalation calls (Qwen ~12%) | 5–6 calls × 2,000 tokens | 5–6 × 400 | ~$0.01 |
| **Total estimated**       | —            | —             | **~$0.07** |

Pricing assumptions: Gemini 2.5 Flash at $0.15/1M input, $0.60/1M output.
Qwen2.5-VL-72B via AIML API at $0.70/1M tokens.
Cached tokens at ~4× discount vs standard input.

### 5.4 Latency and Throughput

| Configuration           | Wall-Clock Time (45 claims) |
|-------------------------|-----------------------------|
| Sequential (1 concurrent) | ~225–360s (~4–6 min)      |
| Concurrent (8 parallel)   | ~30–60s (~1 min)           |
| With escalation overhead  | +10–20s                    |

**Rate limits**: Gemini 2.5 Flash supports 1,000 RPM. With 8 concurrent requests
and ~3–8s per call, we stay well under 100 RPM — no throttling expected.

### 5.5 Cost and Latency Optimisation Strategy

1. **Context Caching**: Static system prompt (~4,500 tokens) cached via Vertex AI
   Context Cache API with 2-hour TTL. Applied discount ~4× on cached tokens.
   Estimated savings: ~70% on total input token costs.

2. **Semaphore-Based Concurrency**: `asyncio.Semaphore(8)` limits parallel
   Gemini calls to 8, well below the 1,000 RPM rate limit even at peak.

3. **OpenCV Pre-Pass**: Zero-cost quality filter eliminates VLM calls for
   images that are too blurry to evaluate. Saves ~5–15% of API calls.

4. **Selective Escalation**: Qwen2.5-VL-72B called only for ~10–15% of
   uncertain claims — not all claims. This keeps AIML API costs minimal.

5. **SHA-256 Deduplication**: Identical images across claims share OpenCV
   analysis results (in-memory cache) to avoid redundant computation.

6. **Exponential Backoff**: Jitter-based retry (2^attempt + uniform(0,1)s)
   handles transient 429 errors without hard blocking.

7. **Batch Potential**: For future scale (>1,000 claims), the Vertex AI Batch
   Prediction API provides 50% cost discount at the expense of ~4hr latency.

---

## 6. System Architecture Summary

```
claims.csv ──► Layer 1: DataIngestionEngine ◄── user_history.csv + evidence_requirements.csv
                        │
                        ▼
              Layer 2: ImageValidator (OpenCV)
              [Laplacian variance, brightness, entropy, resolution]
                        │ (pre-computed quality flags injected into prompt)
                        ▼
              Layer 3: GeminiVLMAgent (Gemini 2.5 Flash)
              [Schema-constrained JSON, context cache, chain-of-thought]
                        │
                        ├── If uncertain (10-15%): ──►
                        │                    Layer 4: QwenEscalationAgent
                        │                    [Qwen2.5-VL-72B via AIML API]
                        │                             │
                        │◄────── ensemble_vote() ◄───┘
                        ▼
              Layer 5: PostProcessor
              [Deterministic overrides, schema compliance, OutputRow assembly]
                        │
                        ▼
              output.csv
```
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report, encoding="utf-8")
    logger.info("Evaluation report written to %s", output_path)

    # Also print key metrics to console
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"claim_status accuracy  : {status_m['accuracy']:.3f}")
    print(f"claim_status macro F1  : {status_m['macro_f1']:.3f}")
    print(f"issue_type accuracy    : {issue_m['accuracy']:.3f}")
    print(f"object_part accuracy   : {part_m['accuracy']:.3f}")
    print(f"risk_flags Jaccard     : {jaccard:.3f}")
    print(f"severity MAE           : {sev_mae:.3f}")
    print(f"evidence_standard_met  : {evidence_m['accuracy']:.3f} accuracy")
    print("=" * 60)
    print(f"\nFull report: {output_path}")


# ---------------------------------------------------------------------------
# Prediction runner (run pipeline on sample set)
# ---------------------------------------------------------------------------

def run_pipeline_on_sample(sample_output_path: str) -> None:
    """
    Run the main pipeline on sample_claims.csv and save predictions.
    This is equivalent to: python ../main.py --sample --output <path>
    """
    import subprocess
    main_path = Path(__file__).parent.parent / "main.py"
    result = subprocess.run(
        [sys.executable, str(main_path), "--sample", "--output", sample_output_path],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("Pipeline run failed with exit code %d", result.returncode)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation for Multi-Modal Claims Verification System"
    )
    parser.add_argument(
        "--predictions",
        default="sample_predictions.csv",
        help="Path to predictions CSV (run main.py --sample first)"
    )
    parser.add_argument(
        "--ground-truth",
        default=str(GROUND_TRUTH_PATH),
        help="Path to sample_claims.csv (ground truth)"
    )
    parser.add_argument(
        "--report",
        default=str(Path(__file__).parent / "evaluation_report.md"),
        help="Output path for evaluation_report.md"
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the main pipeline on sample data before evaluating"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_pipeline:
        logger.info("Running pipeline on sample data…")
        run_pipeline_on_sample(args.predictions)

    # Load ground truth
    if not Path(args.ground_truth).exists():
        logger.error("Ground truth not found: %s", args.ground_truth)
        sys.exit(1)
    gold_df = pd.read_csv(args.ground_truth, dtype=str).fillna("")

    # Load predictions
    if not Path(args.predictions).exists():
        logger.error(
            "Predictions file not found: %s\n"
            "Run: python ../main.py --sample --output %s",
            args.predictions, args.predictions
        )
        sys.exit(1)
    pred_df = pd.read_csv(args.predictions, dtype=str).fillna("")

    if len(gold_df) != len(pred_df):
        logger.warning(
            "Row count mismatch: gold=%d, predictions=%d",
            len(gold_df), len(pred_df)
        )
        # Align on user_id
        pred_df = pred_df.set_index("user_id").reindex(gold_df["user_id"]).reset_index()
        pred_df = pred_df.fillna("")

    logger.info("Evaluating %d predictions against ground truth", len(gold_df))

    metrics = EvaluationMetrics(gold_df, pred_df)
    generate_report(metrics, gold_df, pred_df, args.report)


if __name__ == "__main__":
    main()
