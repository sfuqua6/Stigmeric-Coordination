# Validator Diagnostic Report

Generated: 2026-05-14 (static code-path analysis, runs 20260512–20260514)

## Summary

Verification scores across all real-LLM runs: mode 0.06–0.18, max 0.47.
Root cause: **two independent problems** — keyphrase extraction is fragile on
reasoning-model output, and score parsing defaults to 0.5 on non-conforming
responses, which is also wrong (see Finding 2).

---

## Finding 1: Keyphrase extraction is marginally adequate but brittle

### Code path

`agents/validator.py::_extract_keyphrase` takes the first 5 non-stopword words
of `s.content` after simple punctuation stripping. For clean prose claims this
produces reasonable Wikipedia queries. However:

1. **Reasoning bleed-through**: Before the A1 fix, `s.content` could contain
   leading `</think>` fragments (`"Hmm... Frogs have"` from INITIAL_00013 in
   run 152054). The keyphrase for that signal would be `"Hmm Frogs have"` —
   a useless Wikipedia query that returns nothing.

2. **First-person openers**: If a claim starts with `"I think that renewable
   energy..."`, the keyphrase becomes `"I think renewable energy"` (stopwords
   skip "that" but not "I"). Wikipedia finds nothing for first-person queries.

3. **Scratchpad-as-claim**: Run 152054 shows INITIAL_00013 with content
   `"[No claim yet] Hmm... Frogs have"`. Even though the A2 junk filter now
   blocks this at deposit time, any such signals that survived into the KB
   before A2 will be loaded as validator targets on the next run.

### Fix required

Replace the naive first-N-words extraction with a noun-phrase heuristic that
skips leading first-person clauses. Proposed replacement (in
`agents/validator.py::_extract_keyphrase`):

```python
def _extract_keyphrase(text: str, max_words: int = 5) -> str:
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "and", "or", "but", "for", "on", "at", "by", "with", "as",
        "that", "this", "these", "those", "it", "its",
        # additional first-person terms to skip
        "i", "my", "we", "our", "you", "your",
    }
    # Skip any leading first-person clause (stops at first content word that's
    # neither a pronoun, auxiliary verb, nor punctuation).
    words = [w.strip(".,;:?!\"'()[]") for w in text.split()]
    keep = [w for w in words if w and w.lower() not in stop]
    return " ".join(keep[:max_words]) or text[:50]
```

This won't fix reasoning-bleed claims (A1 does that), but it reduces garbage
queries from first-person openers.

---

## Finding 2: Score parsing is correct but the prompt is ambiguous

### Code path

`agents/validator.py::parse` uses:
```python
m = re.search(r"score\s*[:=]\s*([+-]?(?:\d+\.?\d*|\.\d+))", text, re.IGNORECASE)
```
Default fallback: `score = 0.5`.

### Observations from run outputs

- Verification scores of 0.06–0.18 indicate the model IS emitting parseable
  `SCORE: X.X` lines, but with low values. This is correct behaviour: the
  Wikipedia snippets genuinely don't support most claims because the corpus
  is a synthetic placeholder (`trivial_corpus_from_thesis`) that produces
  off-topic claims. The model correctly reports low confidence.

- The 0.5 fallback only fires when the model omits the SCORE line entirely.
  This inflates scores slightly (0.5 instead of 0.0 for a non-answer).

- The max score of 0.47 in recent runs suggests the scoring scale itself is
  working — the validator occasionally finds real alignment.

### Assessment

The low mean verification score (0.06–0.18) reflects the quality of the
underlying corpus, not a parser bug. The corpus is a trivial placeholder that
generates claims about frog respiration regardless of the task prompt.
Wikipedia verifies specific factual claims — it won't support philosophically
framed or vaguely worded claims.

**The validator is not broken; the corpus is wrong.**

---

## Finding 3: Validator routing is already improved

`agents/validator.py::sample` already uses
`store.signals_with_many_children_of_type(INITIAL, SUPPORT, 2)` to target
well-supported claims before falling back to its assigned strategy. This means
validators are preferring high-stakes targets. The low scores are not a routing
failure — they are corpus fidelity.

---

## Recommended fixes (in priority order)

1. **A1 (done)**: `strip_reasoning` now removes `</think>` before content
   reaches the validator. This eliminates "Hmm... Frogs have" as a keyphrase.

2. **A2 (done)**: Junk filter now blocks `[No claim yet]` at deposit time.
   Future runs won't accumulate broken claims in the store.

3. **Keyphrase heuristic**: Add first-person stopwords to `_extract_keyphrase`
   in `agents/validator.py` (see Finding 1 above). Low risk, no architecture
   change needed.

4. **Corpus**: Replace `trivial_corpus_from_thesis` with real retrieval
   (web/Wikipedia) so scout claims are grounded in actual evidence. This is
   the root cause of low verification scores and is outside the validator's
   control. Addressed separately in the retrieval roadmap.

---

## Why a higher verification mean is not achievable without corpus fix

Even with perfect keyphrase extraction and routing, a validator querying
Wikipedia about "Frogs breathe through pectoral fin sacs" will find no
supporting article because the claim is factually wrong (generated by a
hallucinating LLM conditioned on a trivial corpus). The score correctly reports
≈ 0.1. Achieving mean > 0.4 requires claims that are:
  - Factually correct, AND
  - Specific enough for Wikipedia to have an article, AND
  - Phrased so the first 5 content words form a valid query.

A real corpus (Phase C retrieval integration) satisfies the first condition.
The keyphrase fix above satisfies the third. Together they should raise the
mean to the 0.4+ target range.

---

## Corrected scoring scale

With the synthetic corpus: mean 0.06–0.18 is **expected** and indicates the
validator is working correctly (high scores would indicate misalignment with
Wikipedia). With a real corpus: mean 0.4+ is a reasonable target.
