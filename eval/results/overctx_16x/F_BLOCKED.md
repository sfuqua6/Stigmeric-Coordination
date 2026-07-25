# Condition F (16x pack) — structurally blocked on this Groq account, not a bug

**Status: STOP condition met (pre-registered).** Real F answers were not shipped.
`conditions.jsonl` keeps F rows empty (see `conditions.jsonl.bak-invalid-F`); do
not fabricate or judge against them.

## Root cause (verified against the live API, not inferred)

1. **Per-minute token budget (TPM) for `llama-3.1-8b-instant` on this account
   is 6000 tokens/minute.** Verified directly from live response headers on
   2026-07-24:
   ```
   x-ratelimit-limit-tokens: 6000
   ```

2. **Every one of the six 16x-scale packs requires ~67,000–76,000 real prompt
   tokens** once rendered through `_evidence_from_pack` + `_RAG_DIRECT`
   (measured via a live 429 error body, which reports the API's own tokenizer
   count, not our char/4 estimate):
   ```
   Rate limit reached for model `llama-3.1-8b-instant` ... tokens per day (TPD):
   Limit 500000, Used 493492, Requested 76064.
   ```
   Per-prompt real token counts (measured, using the exact ratio from that
   error): oc_innovation_drivers ~76,071; oc_inequality_conflict ~67,145;
   oc_remote_work_conflict ~76,054; oc_nuclear_debate ~76,068;
   oc_antibiotic_resistance_conflict ~76,069; oc_ubi_evidence ~76,071.
   Every prompt is **11–13x over the per-minute budget on its own**, before a
   single completion token is generated.

3. **Consequence, confirmed empirically:** a request this size is not
   rejected outright — Groq lets a single oversized request through — but the
   completion phase is starved by the same per-minute allowance. The one real
   F completion obtained during this repair
   (`conditions.jsonl.f_progress.json`, `oc_innovation_drivers`) requested
   `max_tokens=1024` and received only **332 completion tokens**, cutting off
   mid-sentence (`"...(In"`). This is not an artifact of a wrong `max_tokens`
   argument or a retriable transient error — it is the per-minute budget
   reasserting itself mid-generation. No amount of retrying or raising
   `max_tokens` changes this outcome for a prompt this size on this account.

4. **Separately, and compounding the problem:** `llama-3.1-8b-instant`'s
   **daily** token budget (TPD) on this account is also nearly exhausted —
   493,492 of 500,000 used as of this session (largely consumed by the prior
   agent's repeated regeneration attempts, each burning ~76k tokens per
   try × up to 10 retries × 6 prompts). A fresh attempt right now 429s
   immediately with 0 tokens generated. Even after the daily reset (~3h20m
   out at time of writing), six full-pack attempts would need
   6 × ~70-76k ≈ 420,000–460,000 tokens for prompts alone — at or above the
   entire daily ceiling, leaving no room for retries, completions, or the
   A/B/E conditions already run that day.

## Why this can't be worked around without breaking the experiment

The task design requires F to consume the **identical** 16x pack condition A
consumes (`--corpus=pack:<path>`) — that's what makes A-vs-F a valid
apples-to-apples over-context comparison. Shrinking F's evidence block to fit
the account's real per-minute budget (~6000 tokens, i.e. ~12x smaller than
even the smallest pack) would no longer be "the same pack," and would
manufacture an artificial A-vs-F delta driven by unequal evidence rather than
consumption strategy — exactly the confound condition F exists to rule out.

## What this means for the kill-criterion

The pre-registered kill criterion (A beats F at 16x, Wilson lower bound > 0.5)
is **underpowered / not evaluable** — not "not met" — because no valid F data
exists at n=6 or any n>0. This is an infrastructure/account-tier limitation of
the free-tier Groq API for this experiment's evidence-pack size, not evidence
about the swarm's compression thesis one way or the other.

## Path forward (not executed here)

- A Groq paid/Dev tier (or any provider with a TPM ceiling well above the
  ~70-76k single-prompt requirement) would remove the per-minute starvation;
  the TPD ceiling would also need enough daily headroom for 6 full attempts.
- Alternatively, run condition F locally (same local GPU model used for the
  swarm's `A` condition under `--backend local`), which has no such per-minute
  token throttle.
