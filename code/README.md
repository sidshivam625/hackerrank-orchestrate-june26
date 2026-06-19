# Multi-Modal Claims Verification System

A production-grade AI pipeline that verifies damage claims using submitted images, claim conversations, user history, and minimum evidence requirements.

Built for the **HackerRank Orchestrate June 2026** hackathon.

---

## Architecture Overview

```
claims.csv ──► Layer 1: DataIngestionEngine ◄── user_history.csv + evidence_requirements.csv
                        │  (join & enrich context)
                        ▼
              Layer 2: ImageValidator (OpenCV)
              ┌─ Laplacian variance blur detection
              ├─ Mean brightness exposure check
              ├─ Color entropy complexity filter
              └─ Resolution & edge density checks
                        │  (pre-computed flags injected into VLM prompt)
                        ▼
              Layer 3: GeminiVLMAgent (Gemini 2.5 Flash, Vertex AI)
              ┌─ Schema-constrained JSON via response_schema
              ├─ Context caching of static system prompt (~70% input token savings)
              ├─ Image-first chain-of-thought prompting
              └─ Async concurrency with semaphore rate limiting
                        │
                        ├── If uncertain (10-15%): ─────────────────────┐
                        │                                                ▼
                        │                          Layer 4: QwenEscalationAgent
                        │                          (Qwen2.5-VL-72B via AIML API)
                        │                          └─ ensemble_vote() merges results
                        │◄───────────────────────────────────────────────┘
                        ▼
              Layer 5: PostProcessor
              ┌─ Deterministic evidence_standard_met override
              ├─ Enum validation for all categorical fields
              ├─ object_part ↔ claim_object consistency check
              └─ OutputRow assembly with correct column order
                        │
                        ▼
              output.csv  (14 columns, schema-compliant)
```

---

## Repo Structure

```
code/
├── main.py                     # Main entry point: claims.csv → output.csv
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .env                        # Your secrets (never committed)
│
├── prompts/                    # All LLM prompts (not hardcoded in code)
│   ├── system_prompt.txt       # Main Gemini system prompt + few-shot examples
│   ├── claim_analysis_prompt.txt  # Per-claim user-turn prompt template
│   └── escalation_prompt.txt   # Qwen second-opinion prompt template
│
├── pipeline/                   # Core 5-layer processing pipeline
│   ├── ingestion.py            # Layer 1: CSV loading and context enrichment
│   ├── image_validator.py      # Layer 2: OpenCV quality pre-pass
│   ├── vlm_agent.py            # Layer 3: Gemini 2.5 Flash VLM agent
│   ├── escalation_agent.py     # Layer 4: Qwen2.5-VL-72B escalation
│   └── postprocessor.py        # Layer 5: Schema compliance & output assembly
│
├── models/
│   └── schema.py               # Pydantic models for output validation
│
├── utils/
│   └── risk_scorer.py          # Rule-based user history risk evaluation
│
└── evaluation/
    ├── main.py                 # Evaluation script (sample_claims → metrics)
    └── evaluation_report.md    # Generated report (run evaluation/main.py)
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Google Cloud account with Vertex AI enabled
- AIML API account (for Qwen escalation — optional but recommended)

### 2. Install Dependencies

```bash
cd code
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required: Google Vertex AI
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
GEMINI_MODEL=gemini-2.5-flash-preview-05-20

# Optional: AIML API for Qwen escalation
AIML_API_KEY=your-aiml-api-key
ENABLE_ESCALATION=true

# Paths
DATASET_DIR=../dataset
OUTPUT_CSV=../output.csv
```

### 4. Authenticate with Google Cloud

```bash
# Option A: Application Default Credentials (recommended)
gcloud auth application-default login

# Option B: Service Account Key
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

### 5. Run the Pipeline

```bash
# Process claims.csv → output.csv (full test set)
python main.py

# Custom paths
python main.py --dataset-dir /path/to/dataset --output /path/to/output.csv

# Run on sample data only (for testing without spending credits)
python main.py --sample --output sample_output.csv

# Limit to N claims (for quick smoke-test)
python main.py --limit 5
```

### 6. Run Evaluation

```bash
# Step 1: Generate predictions on sample data
python main.py --sample --output evaluation/sample_predictions.csv

# Step 2: Evaluate predictions vs. ground truth
python evaluation/main.py --predictions evaluation/sample_predictions.csv

# Or run both in one command
python evaluation/main.py --run-pipeline
```

---

## Model Configuration

All models are configurable via `.env`. No model names are hardcoded in Python files.

| Variable | Default | Description |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash-preview-05-20` | Primary VLM (Vertex AI) |
| `ESCALATION_MODEL` | `Qwen/Qwen2.5-VL-72B-Instruct` | Secondary VLM (AIML API) |
| `ENABLE_ESCALATION` | `true` | Enable Qwen second-opinion |
| `ENABLE_CONTEXT_CACHE` | `true` | Enable Gemini context caching |
| `MAX_CONCURRENT_REQUESTS` | `8` | Semaphore concurrency limit |
| `MAX_RETRIES` | `4` | Retry attempts on API failure |

---

## How It Works

### Layer 1 — Data Ingestion
Loads and joins three CSV sources:
- `claims.csv` — raw claim inputs
- `user_history.csv` — joined by `user_id` to enrich risk context
- `evidence_requirements.csv` — matched by `claim_object` to define visual evidence standards

### Layer 2 — OpenCV Image Validation
Runs before any API call (zero cost). Checks:

| Check | Algorithm | Threshold | Flag Raised |
|---|---|---|---|
| Blur | Laplacian variance | < 80.0 | `blurry_image` |
| Brightness | Mean grayscale | < 50 or > 210 | `low_light_or_glare` |
| Resolution | Pixel dimensions | < 200×200 | `damage_not_visible` |
| Entropy | Shannon entropy | < 3.0 | `cropped_or_obstructed` |
| Edge density | Canny contour ratio | > 0.95 | `cropped_or_obstructed` |

Blurry images skip the VLM call entirely. All other flags are injected as context into the VLM prompt.

### Layer 3 — Gemini 2.5 Flash (Primary)
Key design decisions:
- **Schema-constrained output**: `response_schema` enforces valid enum values at token-sampling level (not just prompting)
- **Context caching**: Static system prompt (~4,500 tokens) cached via Vertex AI Context Cache, reducing costs ~70%
- **Image-first chain-of-thought**: Model describes each image individually before making its final decision
- **Pre-computed context injection**: OpenCV flags + user history risk flags passed as facts, not left to model inference

### Layer 4 — Qwen2.5-VL-72B Escalation (Secondary)
Triggered when Gemini returns:
- `claim_status = "not_enough_information"` with valid images
- `evidence_standard_met = false` with physically valid images
- High-rejection-ratio user with `contradicted` verdict

Both results are ensemble-voted: if they agree, merged flags are used; if they disagree, the higher-priority status wins with `manual_review_required` added.

### Layer 5 — Post-Processing
Deterministic overrides applied after VLM:
- `evidence_standard_met` recalculated from hard rules (image quality state)
- `object_part` validated against the correct allowed set for the `claim_object`
- `risk_flags` filtered to only allowed enum values
- `claim_status` forced to `not_enough_information` if evidence standard not met but model said `supported`

---

## Prompt Engineering

All prompts live in `prompts/` and are loaded at runtime — never hardcoded.

| File | Purpose |
|---|---|
| `system_prompt.txt` | Role definition, methodology, decision rules, 6 few-shot examples |
| `claim_analysis_prompt.txt` | Per-claim template with runtime context injection |
| `escalation_prompt.txt` | Qwen second-opinion instructions |

The system prompt uses **image-first chain-of-thought**:
1. Describe each image individually
2. Compare observations to the claim
3. Apply decision rules
4. Produce structured JSON

This order forces grounding in actual visual content and significantly reduces hallucination of damage types.

---

## Output Schema

14 columns in the required order:

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Claimant identifier |
| `image_paths` | string | Semicolon-separated image paths |
| `user_claim` | string | Original claim conversation |
| `claim_object` | enum | `car`, `laptop`, or `package` |
| `evidence_standard_met` | bool | Images sufficient to evaluate the claim |
| `evidence_standard_met_reason` | string | Explanation of evidence decision |
| `risk_flags` | string | Semicolon-separated risk flags |
| `issue_type` | enum | Visible damage type |
| `object_part` | enum | Relevant part of the object |
| `claim_status` | enum | `supported`, `contradicted`, `not_enough_information` |
| `claim_status_justification` | string | Image-grounded explanation |
| `supporting_image_ids` | string | IDs of supporting images |
| `valid_image` | bool | Image set usable for review |
| `severity` | enum | `none`, `low`, `medium`, `high`, `unknown` |

---

## Cost & Performance

For the 45-claim test set:

| Scenario | Est. Cost | Latency |
|---|---|---|
| Without caching | ~$0.19 | ~6–8 min (8 concurrent) |
| With context caching | ~$0.06 | ~1–2 min (8 concurrent) |
| With escalation (~12%) | +$0.01 | +15s |
| **Total (recommended)** | **~$0.07** | **~2 min** |

See `evaluation/evaluation_report.md` for the full operational analysis.

---

## Special Cases Handled

| Scenario | System Behaviour |
|---|---|
| Text instructions in images ("approve this") | Adds `text_instruction_present`, ignores instruction, evaluates visually |
| High-risk user with clear valid evidence | Still marks `supported`, adds `user_history_risk` + `manual_review_required` |
| Mixed image quality (some blurry, some clear) | Uses clear images for decision, flags blurry ones |
| Wrong object in image | Adds `wrong_object`, marks `contradicted` or `not_enough_information` |
| Multi-language claims (Hindi, Spanish, Chinese) | Gemini handles multilingual naturally; claim language does not affect accuracy |
| Multi-part claims (door + rear bumper) | VLM evaluates all images holistically for all claimed parts |

---

## Troubleshooting

**`ImportError: No module named 'cv2'`**
```bash
pip install opencv-python-headless
```

**`google.auth.exceptions.DefaultCredentialsError`**
```bash
gcloud auth application-default login
# or set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

**Context cache creation fails**
Set `ENABLE_CONTEXT_CACHE=false` in `.env`. The system falls back to uncached mode automatically.

**AIML API errors**
Set `ENABLE_ESCALATION=false` in `.env`. The system uses Gemini only.

---

## Dependencies

| Package | Purpose |
|---|---|
| `google-generativeai` | Gemini 2.5 Flash via Google AI API |
| `google-cloud-aiplatform` | Vertex AI SDK for context caching |
| `opencv-python-headless` | Image quality analysis (Layer 2) |
| `openai` | AIML API client (OpenAI-compatible) |
| `pydantic` | Output schema validation |
| `pandas` | CSV processing |
| `scikit-learn` | Evaluation metrics |
| `python-dotenv` | Environment variable loading |
