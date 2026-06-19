# Evaluation Report
*Generated for Multi-Modal Claims Verification System*

---

## 1. Dataset Overview

| Metric              | Value                  |
|---------------------|------------------------|
| Sample claims       | 20         |
| Claim objects       | car, laptop, package |
| Ground truth source | dataset/sample_claims.csv |

---

## 2. Primary Metric — `claim_status` Performance

**Overall Accuracy: 0.850 | Macro F1: 0.869**

### Per-Class Breakdown

| Class                     | Precision | Recall | F1    | Support |
|---------------------------|-----------|--------|-------|---------|
| supported                 | 0.917     | 0.846  | 0.880 | 13       |
| contradicted              | 0.667     | 0.800  | 0.727 | 5       |
| not_enough_information    | 1.000     | 1.000  | 1.000 | 2       |

### Confusion Matrix

```
Predicted →     supported     contradict    not_enough  
Actual supported   : 11            2             0           
Actual contradict  : 1             4             0           
Actual not_enough  : 0             0             2           
```

---

## 3. Secondary Field Metrics

| Field                   | Metric         | Score    |
|-------------------------|----------------|----------|
| `issue_type`            | Accuracy       | 0.800  |
| `issue_type`            | Weighted F1    | 0.799  |
| `object_part`           | Accuracy       | 0.900  |
| `object_part`           | Weighted F1    | 0.917  |
| `severity`              | Accuracy       | 0.800  |
| `severity`              | Weighted F1    | 0.807  |
| `severity`              | Ordinal MAE    | 0.278  |
| `evidence_standard_met` | Accuracy       | 0.900  |
| `evidence_standard_met` | F1             | 0.941  |
| `valid_image`           | Accuracy       | 0.900  |
| `valid_image`           | F1             | 0.944  |
| `risk_flags`            | Exact Accuracy | 0.400  |
| `risk_flags`            | Macro F1       | 0.477  |
| `risk_flags`            | Jaccard (avg)  | 0.608  |
| `supporting_image_ids`  | Exact Accuracy | 0.850  |
| `supporting_image_ids`  | Macro F1       | 0.867  |

---

## 4. Failure Analysis — Top 5 Worst Cases

The following claims had the most prediction errors:


### Case 1: user_034 (package)
- **Claim snippet**: Customer: My delivery box arrived opened. | Support: Was the package crushed or …
- **Errors**: claim_status: expected=contradicted, got=supported; issue_type: expected=none, got=torn_packaging
- **Predicted justification**: img_1 provides clear visual evidence of a torn seal and damaged cardboard, directly supporting the user's claim. The text instruction on the sticky no

### Case 2: user_001 (car)
- **Claim snippet**: Customer: Hi, I found new damage on my car after it was parked outside overnight…
- **Errors**: claim_status: expected=supported, got=contradicted; issue_type: expected=dent, got=broken_part
- **Predicted justification**: The user claimed a 'dent' on the rear bumper, but img_1 shows catastrophic damage including a missing bumper cover and a severely crushed trunk lid. T

### Case 3: user_002 (car)
- **Claim snippet**: Customer: Parking lot mein meri car ko scrape lag gaya. | Support: Aap kis type …
- **Errors**: claim_status: expected=supported, got=contradicted; issue_type: expected=scratch, got=none
- **Predicted justification**: The user claimed a minor scratch on the front bumper. However, img_2 shows a car with a completely undamaged front bumper, directly contradicting the 

### Case 4: user_008 (car)
- **Claim snippet**: Customer: I picked up my car after service and noticed a mark on the hood. | Sup…
- **Errors**: object_part: expected=front_bumper, got=hood
- **Predicted justification**: The user claimed a minor scratch on the hood, but img_1 shows a completely wrecked vehicle with catastrophic damage to the entire front end, including

### Case 5: user_007 (car)
- **Claim snippet**: Customer: Someone clipped my car while it was parked. | Support: What part was h…
- **Errors**: issue_type: expected=broken_part, got=crack
- **Predicted justification**: img_1 clearly shows the glass of the side mirror is cracked with a spider-web pattern, which directly supports the user's claim of a damaged side mirr


---

## 5. Operational Analysis

### 5.1 Model Call Estimates

| Phase               | Claims | Avg Images/Claim | VLM Calls | Escalation Rate | Total API Calls |
|---------------------|--------|------------------|-----------|-----------------|-----------------|
| Sample evaluation   | 20     | ~2.0             | 20         | ~10-15%         | ~22          |
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
