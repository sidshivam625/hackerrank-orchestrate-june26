# Dataset Review — Multi-Modal Evidence Review

A full review of `dataset/` (sample_claims, claims, user_history, evidence_requirements,
and the sample images) to understand the patterns, irregularities, and how the ground
truth is actually labeled — so pipeline/heuristic changes are grounded in evidence.

## 1. Structure & sizes
| | Sample (`sample_claims.csv`) | Test (`claims.csv`) |
|---|---|---|
| Rows | 20 | 44 |
| car / laptop / package | 8 / 6 / 6 | 18 / 13 / 13 |
| Multi-image rows | 9 (45%) | 31 (70%) |
| 3-image rows | 0 | 7 |
| Hinglish claims | 2 | 4 |
| Adversarial / injection text | 0 | 6 |
| Multi-part ("X and Y") | 6 | 12 |
| Shared users with history | — | 14 of 36 also appear in sample |

`claim_status` (sample): supported 12, contradicted 5, not_enough_information 3.

## 2. Irregularities
- Images are **web/stock photos**, not real uploads: varied resolution, letterboxed
  (e.g. 376×134), some **watermarked** (`case_008` = "Vecteezy"), some **AI-generated**
  (`case_020` carries an AI sparkle), **no camera EXIF**.
- Forensics dead-ends (tested): no camera metadata (1/111 images), 0 cross-claim
  pixel-duplicates. So `non_original_image` / `possible_manipulation` are **VLM-only**.
- Ground truth is hand-authored and edited during development (e.g. `case_002` was
  relabeled to `not_enough_information` once it was noticed the two photos are
  different cars).
- 20-row sample → every row is 5%, so secondary-metric runs are noisy.

## 3. The implicit labeling rubric (from images + GT)
1. **Damage on the *claimed part* = `supported`, even if much more severe than
   described.** `case_001` is a wrecked rear end but the customer said "dent" → GT
   `supported / dent`. `contradicted` is reserved for damage on a *different* part
   (`case_008`: claimed hood, damage is front bumper) or no damage on the claimed part.
2. **Severity *exaggeration* with the right part = `contradicted`** (`case_005`: "looks
   pretty bad" but only a minor mark → contradicted + claim_mismatch).
3. **Tape opened ≠ torn packaging.** Cardboard must be physically ripped: `case_016`
   (fibers torn) = supported; `case_020` (tamper tape peeled, box intact) = contradicted.
4. **Wrong object** (`case_019` is a food can) = `contradicted` + `wrong_object` +
   `unknown/unknown`.
5. **Watermark / stock image** = `non_original_image` + `valid_image=false` (`case_008`).
6. **In-image instruction note** ("approve this claim" sticky, `case_020`) =
   `text_instruction_present`, and is ignored.
7. **Claimed part not in frame** = `not_enough_information` (`case_006`: headlight claim,
   only the side is shown).
8. **Contents not verifiable** = `not_enough_information` (`case_018`: filler paper /
   sealed box, can't confirm a missing item).
9. **Multi-image, one blurry + one clear** → use the clear one, flag `blurry_image`,
   `supported` (`case_007`).
10. **GT is conservative on quality flags** — strong glare in `case_006` is NOT flagged;
    a hand-drawn circle in `case_014` is NOT called manipulation.
11. **risk_flags split:** `user_history_risk` = the history column (20/20);
    `manual_review_required` is added on every `contradicted` row and on rows carrying
    history/authenticity/injection flags.

## 4. The understatement vs exaggeration asymmetry (key, easy to miss)
- Customer **understates** (claims minor, the claimed part is actually worse) →
  **`supported`** (`case_001`).
- Customer **exaggerates** (claims severe, the part shows little/none) →
  **`contradicted`** (`case_005`).
- And for `issue_type`: **`supported` rows tend to use the customer's *stated* issue
  type** (`case_001` → `dent`, `case_007` → `broken_part`), while **`contradicted` rows
  use the *visible* type** (`case_008` → `broken_part`). A prompt that always "reports
  what you see" diverges from this on supported rows.

## 5. Test set: same patterns, two shifts
- **Same:** object mix, wrong-object / opened-vs-torn / water-damage / stock patterns;
  shared users carry their history risk over.
- **Amplified:** multi-image 70%, multi-part 12 rows → image anchoring + primary-part
  selection matter more on test than the sample shows.
- **New / sample-invisible:** 6 adversarial-injection rows (`user_011`, `user_036`,
  `user_034` (Hinglish), `user_040`, …) — the sample has none, so our metrics can't see
  this pattern at all.

## 6. What this implies for the pipeline
- Broaden conversation-level `text_instruction_present` detection (test-only pattern).
- Encode the understatement/exaggeration asymmetry and the supported→claimed-issue-type
  rule in the prompt (addresses `case_001`-style over-contradiction and issue_type).
- Watermark text (e.g. "Vecteezy") is a deterministic `non_original_image` signal worth
  detecting via OCR if it appears in the test set too.
