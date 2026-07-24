# Hey Fable

You're being handed a stigmergic multi-agent LLM pipeline ("the swarm") and asked to help push it forward. Before you do, read this. It's an evidence-first account — grounded in the repo's own output files — of how the swarm has, so far, come up **short of a single direct model call** on the three things that matter: quality, ability, and wall-clock/cost. Every claim below is backed by a direct quote from an artifact in this repo. I'm not editorializing the numbers; I'm quoting them.

## The claim we're testing

From `CLAUDE.md`:

> "**Information partitioning as the diversity engine.** Diversity comes from *what each agent has been shown*, not from prompt or temperature tweaks."

The implicit promise is that many partitioned small-model agents, coordinating through a shared signal store, out-reason a single model. The comparison that would prove it is simple: **swarm(model M) vs one direct call of model M** (and vs a stronger model). That test has now been run. The swarm does not win.

## Bottom line up front

Across every judged comparison that exists in `outputs/kb/` and `eval/results/`, the swarm has **0 wins**. It loses or ties a single call of the *same* 7B model, ties (via judge noise) a 70B, produces answers **3–4× longer** that are mostly internal telemetry, takes **30–91 minutes** where a direct call takes seconds, and its own quality gate fires while its verification score sits near **zero**. There's one genuine ember of promise (below), but on the head-to-head it currently trades ~300× the compute for a worse deliverable.

---

## 1. Quality: it loses or ties the direct model — blind

The delta harness pits the swarm (condition **A**) against a direct call of the same model (**B**), a cheap best-of-5 revise scaffold (**C**), and a stronger 70B (**D**), judged both orderings with Wilson CIs.

From `outputs/kb/cell11.txt` (the largest sample, n=8):

> `| Δ_amp (A vs B) | 8 | 0 | 4 | 4 | 25% | [0.07, 0.59] | no | 272× |`

Zero wins, four ties, four losses. From `outputs/kb/groq3prompt.txt` (n=3):

> `[judge] A_vs_B: win-rate 17% Wilson[0.02,0.69] real_win=False`
> `| ban_cars | tie | no | ... | god_exists | direct | yes | ... | climate_action | direct | yes |`

The direct call won god_exists and climate_action outright. On the newest runs (`report.md`, `report-792ac33a.md`, both Jun 30), every comparison — including **vs the 70B** — came back:

> `| Δ_vs_strong (A vs D) | 2 | 0 | 2 | 0 | 50% | [0.09, 0.91] | no | 269× |`

"All ties" sounds like parity, but the raw `scores.json` shows it's judge noise, not equality — every single verdict flipped when the answer order flipped:

> `"pid": "ban_cars", "winner": "tie", "agreement": false`
> `"pid": "god_exists", "winner": "tie", "agreement": false`

`agreement: false` on 6/6 comparisons means the judge picked whichever answer was shown *first*. That's position bias, not a dead heat — the "ties" carry no quality signal.

**Why the direct answers read better.** The swarm's delivered artifact is dominated by machine exhaust. From `outputs/kb` conditions and older runs, the swarm's final answer includes lines like:

> `- [7] While car-free policies improve local air quality... (rejected: dissent_pressure=19.95 > 1.5)`
> `- [8] Many cities that have implemented car-free zones report improved public health... (held: no verification, no dissent, support_diversity=3 < 4)`

and closes with a raw citation graph:

> `CLAIM  [INITIAL_00012]: While private cars contribute to greenhouse gas emissions...`
> `  Supports:      SUPPORT_00192, SUPPORT_00079, ... Partition: worker_023, worker_010, ...`
> `  support_diversity=19  dissent_pressure=0.15  verification_score=0.12`

On the "does God exist" run it even leaked its own scaffolding and an off-topic claim into the prose:

> "the available evidence suggests that the question defies straightforward empirical resolution... as highlighted by **Brief 1** ([1])"
> `- [12] The membership base of organizations like **AAAI** comprises primarily technologists and researchers focused on machine learning and AI development...`

(An AI-research-org claim, rendered into a theology answer.)

An older DeepSeek-7B run leaked chain-of-thought straight into the final answer (`outputs/latest_output_good_for_7B/answer.txt`):

> "But wait, how does this tie into the claim? The supporting evidence is about the link between human activity and temperature increase..."
> "But wait, the original claim doesn't mention accountability mechanisms directly..."

and a "does God exist" run leaked the *prompt template* itself:

> "Now, build on this by providing two paragraphs: one arguing in favor of the thesis (God exists), using the cosmological argument as evidence; another refuting it with an opposing viewpoint."

By contrast the direct 7B answer to the same prompt is clean and correctly attributed:

> "**Cosmological Argument**: ... Philosophers like Thomas Aquinas articulated this idea, known as the 'Five Ways'... **Teleological Argument**: ... William Paley used the analogy of finding a watch on a heath..."

and the 70B brought concrete grounding the swarm never did:

> "The transportation sector, which includes private cars, accounts for approximately 23% of global greenhouse gas emissions... cities like Copenhagen, Denmark, and Vancouver, Canada, have implemented car-free zones..."

## 2. Ability: the mechanisms don't do what they claim

**Verification is ~0.** The whole "field pressure / validated claims" story is hollow in practice. From the real-run `summary.json` files:

> `outputs/kb/groqrun1.txt` → `"avg_verification_score": 0.0183`
> `interesting_no_facts_massive_output_in_comp/summary.json` → `"avg_verification_score": 0.0547`
> `collab_weak.txt` → `"max_verification_score": 0.0, "avg_verification_score": 0.0`

Yet the pipeline still reports `"quality_met": true` — because the non-factual quality gate keys on cluster counts, not grounding.

**"Diversity" is inverted — the field produces near-duplicates.** `self_bleu` (lexical self-similarity; lower = more diverse) runs high:

> `groqrun1.txt` → `"self_bleu": 0.6208`  ·  `groq_output.txt` → `"self_bleu": 0.6948`

And the "surviving clusters" are the same sentence reworded. From `groqrun1.txt`'s final answer, clusters [1]–[11] are all variants of:

> "Banning private cars in urban areas is being considered by several cities as a strategic move to reduce their contribution to carbon emissions..."

The system detects this and prints it about itself:

> "[INTER-CLUSTER CONTRADICTION] Clusters [INITIAL_00237] and [INITIAL_00048] share topic content and share dissent signals (). These may be **two framings of the same contested claim**."

**The headline diversity metric is a counting artifact.** `groq_output.txt` reports a "strongest" cluster with:

> `support_diversity=62`

That 62 is the number of *workers* that touched it (each worker is stamped as its own partition), not the number of distinct viewpoints — inflated by design.

**It never actually converges.** Almost every substantive run ends by hitting the clock, not by reaching quality:

> `groqrun1.txt` → `"convergence_reason": "cap_time"`
> `lastgroqrun.txt` → `"convergence_reason": "cap_time"`

and the field stays fragmented at the end. From `groqrun1.txt`'s answer:

> "Of 69 claim cluster(s) projected from the signal field: 8 survived field pressure, 0 contested, 9 held as unverified, 1 rejected, **51 weakly supported**."

51 of 69 clusters weakly supported, 0 contested — that's spray, not consensus.

## 3. Wall clock and cost: seconds vs an hour, at ~300×

A single answer costs the swarm 30–91 minutes. From the run metadata:

> `latest_output_good_for_7B/summary.json` → `"total_elapsed_s": 5475.6` (91 min), `"total_llm_calls": 222`
> `interesting_no_facts_massive_output_in_comp/summary.json` → `"total_elapsed_s": 4610.3` (77 min), `"total_llm_calls": 217`
> `groqrun1.txt` → `"wall_clock_s": 1816.5, "total_iterations": 297`

The judge reports the compute multiple against a single call directly:

> `Δ_amp (A vs B) ... | 390× |` (`groq3prompt.txt`)
> `Δ_vs_simple (A vs C) ... | 127× |` and `Δ_amp (A vs B) ... | 382× |` (`report.md`)
> `Δ_vs_strong (A vs D) ... | 269× |` (`report-792ac33a.md`)

And you pay that to produce a **longer** artifact. Word counts from `eval/results/.../conditions.jsonl`:

| prompt | swarm (A) | direct-7B (B) | best-of-5 (C) | 70B (D) |
|---|--:|--:|--:|--:|
| ban_cars | **1,681** | 543 | 393 | 669 |
| god_exists | **1,828** | 452 | 491 | 899 |

3–4× the length, most of it the census/citations dump quoted in §1. Even the 70B's `god_exists` answer — verbose for a direct call — was less than half the swarm's, and it truncated mid-sentence ("Some people understand God as a"), which tells you the token budget was being spent on length, not completion.

---

## In fairness (so you don't over-correct)

Two honest caveats, and one real ember:

- **The samples are tiny.** n = 2, 3, 8. The Wilson intervals are enormous (`[0.09, 0.91]`). "The swarm loses" is directionally strong (0 wins across ~16 comparisons) but not statistically airtight. Don't treat any single verdict as gospel.
- **The judge is the weak link, not just the swarm.** `agreement: false` on every Jun-30 comparison means the judge is position-biased and length-seduced (its stock phrase is "more comprehensive and nuanced"). A neutral, length-normalized judge on ≥20 prompts is required before *any* of these verdicts — win or loss — is trustworthy.
- **There is a genuine ember.** Judged on *reasoning substance* (Section 1 only, stripping the exhaust), the swarm's `ban_cars` answer was the **best-framed of the four** — the only one to surface context-dependence (small vs large cities) and the second-order displacement effect (bans push people to ride-hailing, raising congestion). That's the orchestration doing what it's supposed to: exploring the answer-space and synthesizing a sharper frame. It did **not** show up on the canonical `god_exists` prompt, where a competent model already has the structure. So the plausible niche is **open-ended, contested, multi-stakeholder questions**, not textbook ones.

## What this means for you, Fable

The shortfall is real but it's concentrated in two fixable places, not in the core idea:
1. **Delivery.** The swarm's actual argument is competitive; it's buried under ~1,000–1,400 words of `support_diversity=`/`PROCESS NOTES`/citation-graph exhaust plus leaked scaffolding. Ship Section 1 only (see `SYNTH_CLEAN_ANSWER_BRIEF.md`). That alone moves it from "obviously worst artifact" to "competitive on quality."
2. **Measurement.** Fix the judge (neutral family, length-normalized, ≥20 prompts, populate the factual ground-truth set) before trusting any delta. Add the missing control: a **single call given the same "synthesize the competing positions" instruction** — if that matches the swarm's Section 1, the value was the prompt, not the orchestration.

Then, and only then, is the honest question worth asking: is the swarm's quality edge on contested questions large and reliable enough to justify **127–382×** the cost? On today's evidence the answer is no. The job is to find out whether that's a ceiling or a starting line — and to let the numbers, quoted like the ones above, decide.

---

### Sources (all in-repo)
- `CLAUDE.md` — design claims
- `outputs/latest_output_good_for_7B/{summary.json,answer.txt}`
- `outputs/interesting_no_facts_massive_output_in_comp/{summary.json,answer.txt}`
- `outputs/kb/{groqrun1.txt,lastgroqrun.txt,groq_output.txt,collab_weak.txt,groq3prompt.txt,cell11.txt}`
- `eval/results/.../{report.md,scores.json,conditions.jsonl}` (and the uploaded `report-792ac33a.md`, `scores-7d2ff592.json`, `conditions-9fb75998.jsonl`)
- `SYNTH_CLEAN_ANSWER_BRIEF.md`, `CLAUDE_CODE_FIX_BRIEF.md` — the fix plans
