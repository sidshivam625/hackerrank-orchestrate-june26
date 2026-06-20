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

**Overall Accuracy: 0.850 | Macro F1: 0.813**

### Per-Class Breakdown

| Class                     | Precision | Recall | F1    | Support |
|---------------------------|-----------|--------|-------|---------|
| supported                 | 0.800     | 1.000  | 0.889 | 12       |
| contradicted              | 1.000     | 0.600  | 0.750 | 5       |
| not_enough_information    | 1.000     | 0.667  | 0.800 | 3       |

### Confusion Matrix

```
Predicted →     supported     contradict    not_enough  
Actual supported   : 12            0             0           
Actual contradict  : 2             3             0           
Actual not_enough  : 1             0             2           
```

---

## 3. Secondary Field Metrics

| Field                   | Metric         | Score    |
|-------------------------|----------------|----------|
| `issue_type`            | Accuracy       | 0.800  |
| `issue_type`            | Weighted F1    | 0.782  |
| `object_part`           | Accuracy       | 0.900  |
| `object_part`           | Weighted F1    | 0.917  |
| `severity`              | Accuracy       | 0.750  |
| `severity`              | Weighted F1    | 0.760  |
| `severity`              | Ordinal MAE    | 0.294  |
| `evidence_standard_met` | Accuracy       | 0.950  |
| `evidence_standard_met` | F1             | 0.971  |
| `valid_image`           | Accuracy       | 0.900  |
| `valid_image`           | F1             | 0.941  |
| `risk_flags`            | Exact Accuracy | 0.550  |
| `risk_flags`            | Macro F1       | 0.668  |
| `risk_flags`            | Jaccard (avg)  | 0.731  |
| `supporting_image_ids`  | Exact Accuracy | 0.800  |
| `supporting_image_ids`  | Macro F1       | 0.770  |

---

## 4. Failure Analysis — Top 5 Worst Cases

The following claims had the most prediction errors:


### Case 1: user_034 (package)
- **Claim snippet**: Customer: My delivery box arrived opened. | Support: Was the package crushed or …
- **Errors**: claim_status: expected=contradicted, got=supported; issue_type: expected=none, got=torn_packaging
- **Predicted justification**: img_1 provides clear evidence of a torn seal and damaged cardboard, supporting the user's claim of a torn-open package. The contradictory evidence in 

### Case 2: user_008 (car)
- **Claim snippet**: Customer: I picked up my car after service and noticed a mark on the hood. | Sup…
- **Errors**: claim_status: expected=contradicted, got=supported; object_part: expected=front_bumper, got=hood
- **Predicted justification**: The user claimed a scratch on the hood. The provided image shows the car's hood is severely damaged as part of a major front-end collision. Although t

### Case 3: user_002 (car)
- **Claim snippet**: Customer: Parking lot mein meri car ko scrape lag gaya. | Support: Aap kis type …
- **Errors**: claim_status: expected=not_enough_information, got=supported
- **Predicted justification**: The user claimed a scratch on the front bumper. While img_1 shows damage far more severe than a scratch (a crushed and broken bumper), the claim is su

### Case 4: user_007 (car)
- **Claim snippet**: Customer: Someone clipped my car while it was parked. | Support: What part was h…
- **Errors**: issue_type: expected=broken_part, got=crack
- **Predicted justification**: The user claimed the side mirror was damaged, and img_1 clearly shows the mirror glass is cracked, which supports the claim.

### Case 5: user_001 (car)
- **Claim snippet**: Customer: Hi, I found new damage on my car after it was parked outside overnight…
- **Errors**: issue_type: expected=dent, got=broken_part
- **Predicted justification**: The user claimed a dent in the rear bumper area. img_1 shows this area has sustained severe damage, including a missing bumper cover and a crushed tru


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
