This file is the **brain of the system**. Everything before this point (ingestion + OpenCV) is just preparing context. This is where Gemini actually looks at the images and decides:

```text
Is the claim supported?
What damage exists?
Which image proves it?
What risks are present?
```

The architecture is surprisingly mature for a hackathon project. It has schema enforcement, caching, retries, self-consistency voting, rate limiting, and ensemble support. 

---

# Full Layer 3 Flow

```text
ClaimContext
     ↓
Risk Scorer
     ↓
Prompt Builder
     ↓
Attach Images
     ↓
Gemini 2.5 Flash
     ↓
JSON Schema Enforcement
     ↓
Pydantic Validation
     ↓
Risk Flag Merge
     ↓
ClaimAnalysisResult
```

---

# 1. Why Gemini is the Primary Agent

The system intentionally uses:

```python
gemini-2.5-flash-preview-05-20
```

as the first reviewer. 

Reason:

```text
Fast
Cheap
Strong multimodal reasoning
Structured output support
```

Qwen is only a backup.

Gemini handles:

```text
90-95% of claims
```

directly.

---

# 2. Context Caching

One of the biggest optimizations.

The system prompt:

```python
self._system_prompt
```

is large and reused for every claim.

Instead of sending:

```text
10,000 token instructions
```

every request:

```python
CachedContent.create(...)
```

stores it once. 

Then later requests only send:

```text
User claim
Image data
Runtime context
```

---

Example:

Without cache:

```text
System Prompt = 10k tokens
Claim Prompt  = 1k tokens

Total = 11k
```

100 claims:

```text
1.1 million tokens
```

---

With cache:

```text
System prompt stored once
```

Each claim:

```text
~1k tokens
```

Huge savings.

---

# 3. Vertex AI vs AI Studio Auto Selection

The code automatically decides:

```python
if GCP_PROJECT_ID exists:
    use Vertex AI
else:
    use AI Studio
```

So the exact same code works for:

```text
Hackathon deployment
Local testing
Production GCP
```

without modification. 

---

# 4. Semaphore-Based Concurrency

This prevents API overload.

```python
self._semaphore = asyncio.Semaphore(
    max_concurrent
)
```

Default:

```python
max_concurrent = 8
```

Meaning:

```text
Claim 1
Claim 2
...
Claim 8
```

can run simultaneously.

Claim 9 waits.

---

Without this:

```text
100 claims
↓
100 simultaneous Gemini calls
↓
429 errors
```

---

With semaphore:

```text
Only 8 active at a time
```

Much safer.

---

# 5. Self-Consistency Sampling

This is a very interesting heuristic.

Normally:

```python
self_consistency_samples = 1
```

Meaning:

```text
One Gemini call
```

---

But if:

```python
self_consistency_samples = 5
```

Then:

```text
Gemini Run #1
Gemini Run #2
Gemini Run #3
Gemini Run #4
Gemini Run #5
```

all analyze the same claim. 

---

Example:

Outputs:

```text
supported
supported
contradicted
supported
supported
```

Majority vote:

```text
supported
```

wins.

This reduces stochastic errors.

---

# 6. Prompt Construction

The VLM doesn't only receive images.

The prompt includes:

### User Claim

```text
My laptop screen cracked.
```

### Claim Object

```text
laptop
```

### OpenCV Quality Flags

```text
blurry_image
low_light_or_glare
```

### User History

```text
past_claim_count=15
rejected_claim=9
```

### Evidence Rules

```text
Need at least one close-up image
```

All are inserted into:

```python
_build_user_prompt()
```

before Gemini sees anything. 

---

# 7. User Risk Scoring

Before Gemini runs:

```python
compute_user_risk_flags(...)
```

creates risk indicators.

Example:

```text
rejection_ratio = 0.7
```

might become:

```text
user_history_risk
```

This gets injected into the prompt.

Gemini therefore knows:

```text
This user historically submits many rejected claims.
```

---

# 8. Prompt Hardening

This is a security feature.

User claims are truncated:

```python
MAX_CLAIM_CHARS = 6000
```

If someone submits:

```text
100,000 character prompt injection
```

the system reduces it to:

```text
6000 chars
```

plus:

```text
...[truncated]
```

This prevents:

```text
Huge costs
Context overflow
Prompt stuffing attacks
```

---

# 9. Multi-Image Grounding

Every image is anchored.

Instead of:

```python
image1
image2
```

Gemini receives:

```text
=== IMAGE img_1 ===
<image>

=== IMAGE img_2 ===
<image>
```



This enables:

```json
{
  "supporting_image_ids":"img_2"
}
```

later.

Without anchoring, image attribution becomes unreliable.

---

# 10. Response Schema Enforcement

This is arguably the strongest design decision.

Most projects do:

```text
Please return JSON.
```

and hope the model obeys.

This project uses:

```python
response_schema=
GEMINI_RESPONSE_SCHEMA
```

inside generation config. 

That means Gemini is constrained during generation.

Not after.

During generation.

---

Instead of:

```json
{
 "status":"probably true"
}
```

Gemini is forced into allowed schema structure.

Much more reliable.

---

# 11. JSON Output Only

The model is configured:

```python
response_mime_type =
"application/json"
```

So Gemini cannot return:

```text
The claim appears supported...
```

It must return JSON.

---

# 12. Pydantic Validation

After Gemini returns:

```python
data = json.loads(...)
```

they create:

```python
ClaimAnalysisResult(**data)
```

This validates:

```text
Enums
Types
Required fields
Formats
```

---

Example:

Gemini returns:

```json
{
 "severity":"very_high"
}
```

Pydantic rejects it.

Validation fails.

Retry triggered.

---

# 13. Automatic Retry System

The entire model call is wrapped inside:

```python
_analyse_with_retry()
```

Default:

```python
max_retries = 4
```

---

## Case A: Rate Limit

If:

```text
429
```

occurs:

```python
wait =
2^attempt + random
```

Exponential backoff.

Example:

```text
1s
2s
4s
8s
```

---

## Case B: Schema Failure

If Pydantic fails:

The error message is fed back into Gemini:

```text
PREVIOUS ATTEMPT FAILED SCHEMA VALIDATION
```

and Gemini tries again. 

This dramatically improves structured-output reliability.

---

# 14. Risk Flag Ownership Model

One of the smartest pieces.

OpenCV owns:

```text
blurry_image
cropped_or_obstructed
low_light_or_glare
```

User-history layer owns:

```text
user_history_risk
```

Gemini is forbidden from generating them.

```python
forbidden_model_flags
```

removes those if Gemini tries. 

Why?

Because those flags already come from deterministic systems.

---

Instead Gemini focuses on:

```text
wrong_object
claim_mismatch
possible_manipulation
damage_not_visible
```

which require visual reasoning.

---

# 15. Fallback Result

If everything fails:

```text
API outage
Bad JSON
Repeated failures
```

the system never crashes.

It returns:

```json
{
 "claim_status":"not_enough_information",
 "risk_flags":"manual_review_required"
}
```



Safe failure mode.

---

# 16. Self-Consistency Voting

If multiple Gemini samples exist:

```text
supported
supported
contradicted
supported
NEI
```

The voter decides.

---

## Claim Status Vote

Majority wins.

```text
supported
```

---

## Tie Break

Order:

```python
NEI < contradicted < supported
```

Meaning:

```text
When unsure,
prefer abstaining.
```

This is a conservative insurance-review philosophy. 

---

## Risk Flag Vote

A flag survives only if:

```text
Majority of samples agree.
```

Example:

```text
possible_manipulation
```

appears once:

```text
discarded
```

appears 3/5 times:

```text
kept
```

This filters hallucinated flags.

---

# Why This VLM Layer Is Strong

Compared to a typical hackathon implementation:

```text
Image
 ↓
Gemini
 ↓
JSON
```

this design adds:

```text
OpenCV quality context
+
User history context
+
Evidence requirements
+
Schema constrained decoding
+
Pydantic validation
+
Automatic retries
+
Context caching
+
Concurrency control
+
Self-consistency voting
+
Risk flag ownership
+
Safe fallbacks
```

So Layer 3 is not just "call Gemini". It's a fairly complete multimodal decision engine whose primary responsibility is to produce a structured `ClaimAnalysisResult` that later layers (escalation and post-processing) can trust. 
