# Amplification-delta report — `overctx_16x`

- model M (A/B/C): `llama-3.1-8b-instant`
- strong model M+ (D): `None`
- conditions: A, B, E, F
- judge: `groq:openai/gpt-oss-120b`
- prompts: 6 (pre-registered: oc_innovation_drivers, oc_inequality_conflict, oc_remote_work_conflict, oc_nuclear_debate, oc_antibiotic_resistance_conflict, oc_ubi_evidence)
- scaffold (C): revise

## Deltas (A = swarm is the focus condition)

| Comparison | n | win | tie | loss | win-rate | Wilson 95% | real win? | cost× |
|---|--:|--:|--:|--:|--:|---|:--:|--:|
| Δ_amp (A vs B) | 6 | 0 | 0 | 6 | 0% | [0.00, 0.39] | no | 129× |
| Δ_vs_prompt (A vs E) | 6 | 1 | 0 | 5 | 17% | [0.03, 0.56] | no | 129× |

> A **real win** = Wilson lower bound clearly above 50%. Read the win-rate next to the cost multiple: a small win at a large cost multiple is a practical loss.

## Objective factual score (judge-free anchor)

Mean fraction of ground-truth checklist items present, over factual prompts.

| Condition | factual score |
|---|--:|
| A — swarm(M) | 81% |
| B — direct(M) | 100% |
| E — synth-prompt(M) | 79% |
| F — rag(M) | 0% |

## Per-prompt verdicts

### Δ_amp (A vs B)

| prompt | winner | agreed orders? | rationale |
|---|:--:|:--:|---|
| oc_innovation_drivers | direct | yes | Answer 2 directly lists the most consistent innovation drivers, details contested factors, and explains sources of disag |
| oc_inequality_conflict | direct | yes | Answer 2 presents a clear, well‑structured synthesis of the evidence, identifies multiple drivers (union decline, tax po |
| oc_remote_work_conflict | direct | yes | Answer 2 directly summarizes the evidence on productivity changes and clearly outlines where sources disagree, while Ans |
| oc_nuclear_debate | direct | yes | Answer 2 directly addresses the question, clearly identifies conflicting claims, and explains how it weighed them, offer |
| oc_antibiotic_resistance_conflict | direct | yes | Answer 2 directly cites the specific evidence sources, details concrete recommendations, and explicitly outlines contrad |
| oc_ubi_evidence | direct | yes | Answer 2 provides a clear, well‑structured synthesis of pros and cons, identifies concrete points of disagreement, and s |

### Δ_vs_prompt (A vs E)

| prompt | winner | agreed orders? | rationale |
|---|:--:|:--:|---|
| oc_innovation_drivers | synthprompt | yes | Answer 2 directly identifies education/research as the most consistent driver, cites evidence, and clearly outlines wher |
| oc_inequality_conflict | synthprompt | yes | Answer 2 provides a clear, coherent synthesis of the evidence, identifies the key drivers, and explicitly notes the clai |
| oc_remote_work_conflict | synthprompt | yes | Answer 2 presents a clear, well‑structured synthesis of the evidence, explicitly noting both positive and negative findi |
| oc_nuclear_debate | synthprompt | yes | Answer 2 directly states a conditional conclusion, clearly identifies conflicting claims, and explains how it weighed th |
| oc_antibiotic_resistance_conflict | synthprompt | yes | Answer 2 cites specific study evidence, details contradictions, and offers nuanced, well‑structured recommendations, whe |
| oc_ubi_evidence | swarm | yes | Answer 1 directly attempts to identify specific source disagreements and the evidence needed to resolve them, fulfilling |

## Delta_vs_rag (A vs F) -- BLOCKED, not evaluated

Condition F could not be regenerated: this Groq account's per-minute token
budget for `llama-3.1-8b-instant` (6000 TPM, verified via live response
headers) is 11-13x smaller than a single 16x-pack prompt (~67k-76k real
tokens, verified via a live 429 error body), so any completion is starved to
a few hundred tokens regardless of `max_tokens` -- confirmed empirically (one
real attempt requested 1024 completion tokens, received 332, cut off
mid-sentence). The account's daily budget for this model is also nearly
exhausted (493,492/500,000 used). See `F_BLOCKED.md` in this directory for
full detail. The pre-registered kill criterion for A-vs-F is underpowered /
not evaluable at n=6 on this infrastructure -- not "met" or "not met."

Order-agreement (both judge orders agreed, no position-bias tie): A_vs_B
6/6, A_vs_E 6/6.
