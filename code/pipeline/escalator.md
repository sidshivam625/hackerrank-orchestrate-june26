The escalation agent is essentially a **"second opinion VLM"**. Gemini is the primary reviewer. Qwen is only called when the system thinks Gemini is uncertain or the claim is risky. 

# Overall Flow

```text
Claim + Images
      ↓
Gemini Analysis
      ↓
_should_escalate() ?
      ↓
   YES
      ↓
Qwen Analysis
      ↓
ensemble_vote()
      ↓
Final Result
```

---

# Step 1: When Does Escalation Trigger?

The decision is made inside:

```python
_should_escalate(...)
```

There are only **3 actual trigger conditions**.

---

## Trigger 1: Gemini says "Not Enough Information"

Condition:

```python
if (
    primary_result.claim_status == "not_enough_information"
    and primary_result.valid_image
):
    return True
```

Meaning:

```text
Images exist
Images passed quality checks
But Gemini could not decide
```

Example:

User:

```text
My windshield is cracked
```

Image:

```text
Windshield visible
Crack partially visible
```

Gemini outputs:

```json
{
  "claim_status": "not_enough_information"
}
```

Since images are valid, system asks:

```text
Let's get Qwen's opinion.
```

This is the most important escalation path. 

---

## Trigger 2: Evidence Standard Failed Despite Valid Images

Condition:

```python
if not primary_result.evidence_standard_met
   and primary_result.valid_image:
```

Example:

Gemini says:

```json
{
  "evidence_standard_met": false,
  "valid_image": true
}
```

This means:

```text
Image quality is okay
But Gemini thinks evidence is insufficient
```

The system suspects uncertainty and asks Qwen.

---

## Trigger 3: Contradicted Claim From High-Risk User

Condition:

```python
if (
    primary_result.claim_status == "contradicted"
    and ctx.user_history
    and ctx.user_history.rejection_ratio > 0.5
):
```

Meaning:

```text
User has many previously rejected claims
Gemini says current claim is contradicted
```

Example:

```text
Past claims:
10 claims
7 rejected

rejection_ratio = 0.7
```

Current claim:

```json
{
   "claim_status": "contradicted"
}
```

System performs a second review.

Reason:

```text
High-risk users generate many false claims,
so important contradictions should be double checked.
```

---

# When Escalation DOES NOT Trigger

---

## No Valid Images

Condition:

```python
if ctx.valid_images_count == 0:
    return False
```

Example:

```text
All images blurry
All images corrupted
No image uploaded
```

Then:

```text
Don't waste Qwen call.
```

Qwen cannot recover missing evidence. 

---

## Escalation Disabled

```python
enable_escalation=False
```

Then:

```python
return False
```

Useful for:

```text
Cost-saving mode
Offline evaluation
Ablation experiments
```

---

# Step 2: What Happens During Escalation?

When escalation happens:

```python
analyse_claim_async()
```

calls:

```python
_call_qwen(...)
```

---

## Prompt Construction

Qwen receives:

### User Claim

```text
My laptop screen cracked
```

### Claim Object

```text
laptop
```

### Gemini Verdict

```text
not_enough_information
```

### Gemini Risk Flags

```text
damage_not_visible
```

### Evidence Requirements

Something like:

```text
screen:
at least 1 image showing crack
```

All are inserted into:

```python
escalation_prompt.txt
```

before sending to Qwen. 

---

## Image Processing Before Qwen

Every image is:

### Converted to RGB

```python
pil_img.convert("RGB")
```

Fixes:

```text
CMYK JPEG
Palette PNG
Progressive JPEG
```

---

### Resized

```python
MAX_DIM = 1568
```

Large phone images:

```text
4032 × 3024
```

become:

```text
1568 × 1176
```

Purpose:

```text
Reduce API payload size
Avoid AIML upload limits
```

---

### JPEG Compression

```python
quality = 85
```

Purpose:

```text
Lower bandwidth
Faster upload
```

---

## Image Anchoring

Before every image:

```text
=== IMAGE img_001 ===
```

is inserted.

Why?

Suppose:

```text
Image 1 shows crack
Image 2 shows scratch
```

Qwen can later say:

```json
{
  "supporting_image_ids": "img_001"
}
```

instead of hallucinating image references.

---

# Step 3: Qwen Inference

Model:

```python
alibaba/qwen3-vl-32b-instruct
```

through AIML API.

Settings:

```python
temperature = 0.0
```

This is important.

No randomness.

Same input should produce nearly same output.

---

# Step 4: Ensemble Voting

This is where the interesting logic lives.

Function:

```python
ensemble_vote(...)
```

---

# Case 1: Both Models Agree

Example:

Gemini:

```text
supported
```

Qwen:

```text
supported
```

Result:

```text
supported
```

Risk flags merged.

Example:

Gemini:

```text
blurry_image
```

Qwen:

```text
wrong_angle
```

Final:

```text
blurry_image;wrong_angle
```

---

# Case 2: Gemini Was NEI

Example:

Gemini:

```text
not_enough_information
```

Qwen:

```text
supported
```

Then:

```python
secondary wins
```

Result:

```text
supported
```

plus:

```text
manual_review_required
```

Why?

The entire purpose of escalation is:

```text
Resolve Gemini uncertainty.
```

If Gemini abstains and Qwen is decisive, trust Qwen. 

---

# Case 3: Gemini Was Decisive

Example:

Gemini:

```text
supported
```

Qwen:

```text
contradicted
```

Result:

```text
KEEP GEMINI
```

and add:

```text
manual_review_required
```

This is a deliberate design choice.

Old version apparently did:

```text
Prefer supported
```

which increased false positives.

New logic:

```text
Never override a decisive Gemini verdict.
```

Instead:

```text
Flag disagreement.
```

Human reviews it.

---

# Example End-to-End

Suppose:

### User

```text
My bumper is cracked
```

### Gemini

```json
{
  "claim_status":"not_enough_information",
  "valid_image":true,
  "risk_flags":"wrong_angle"
}
```

Escalation triggers because:

```text
NEI + valid image
```

---

### Qwen

```json
{
  "claim_status":"supported",
  "risk_flags":"none"
}
```

---

### Ensemble

Since Gemini abstained:

```text
Adopt Qwen verdict
```

Final:

```json
{
  "claim_status":"supported",
  "risk_flags":"manual_review_required;wrong_angle"
}
```

---

# In One Sentence

The escalation agent is a **cost-controlled second-opinion system** that only calls Qwen when Gemini is uncertain, evidence is borderline, or a high-risk contradiction occurs; Qwen can resolve uncertainty, but it generally cannot override a decisive Gemini verdict and instead forces human review when the two models disagree. 
