"""Stock-specific agent roles (task_type="stock").

Built on the same pattern as agents/coding_roles.py: domain-specialised role
classes registered via core/role_registry.py. Where coding verifies "does the
code parse / pass tests", stock verifies "does the numeric claim match
ground-truth market data" (core/stock_verify.py).

IMPLEMENTATION STATUS (see STOCK_SWARM_POC_PROMPT.md Part 5):
  REAL (correctness-critical, the parts that must not be silently wrong):
    * DataValidator   — numeric VERIFICATION via stock_verify.verify_claim
    * ValuationCritic — CRITIQUE_POSITIVE/NEGATIVE from numeric consistency
    * LensScout       — lens partitioning + domain junk gate (needs a number)
    * EquityBriefSynthesizer.build_prediction() — the gradable artifact
  SCAFFOLD (TODO(stage-3)/TODO(refine) markers in-line):
    * ThesisDeveloper / RiskHater prompt wording polish
    * EquityBriefSynthesizer prose rendering (currently deterministic, LLM-free)
    * predicted_return_pct mapping (transparent first-cut heuristic; tune on the
      historical DB before trusting the magnitude)

The no-leak rule and partition invariant apply unchanged: scouts/developers
deposit with partition_id (the lens); all prompts render only signal.content.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

from agents.base import BaseAgent, AgentRunStats, strip_reasoning, type_parent_instruction
from agents.developer import Developer
from agents.synthesizer import Synthesizer
from core.signal_store import SignalStore, Signal
from core.signal_types import (
    INITIAL, SUPPORT, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE, OBJECTION, VERIFICATION,
)
from core.config import (
    MAX_TOKENS_SCOUT, MAX_TOKENS_FORAGER, MAX_TOKENS_CRITIC,
    MAX_TOKENS_HATER, MAX_TOKENS_VALIDATOR,
)
from core.diversity import AgentContextRecord
from core.sampling import SamplingStrategy
from core import stock_verify
from core.stock_data import Snapshot, LENSES


def _has_number(text: str) -> bool:
    return any(ch.isdigit() for ch in (text or ""))


# ---------------------------------------------------------------------------
# LensScout — one analytical lens, disjoint facts (REAL)
# ---------------------------------------------------------------------------

class LensScout(BaseAgent):
    """Scout assigned ONE analytical lens; sees only that lens's facts.

    The lens name is the partition_id (the stock analog of a disjoint corpus
    slice). Every INITIAL must reference a number — a stock claim with no
    metric is junk (the D3 fix: "the artifact presents..." meta-commentary
    cannot survive this gate).
    """
    ROLE = "scout"
    OUTPUT_TYPE = INITIAL
    INPUT_TYPE = None
    MAX_TOKENS = MAX_TOKENS_SCOUT
    TEMPERATURE = 0.8
    DEFAULT_DEPOSIT_STRENGTH = 0.55

    def __init__(self, agent_id: str, llm, snapshot: Snapshot, lens: str,
                 task_prompt: str, **_):
        super().__init__(agent_id, llm)
        self.snapshot = snapshot
        self.lens = lens
        self.task_prompt = task_prompt

    def sample(self, store: SignalStore) -> list[Signal]:
        return []

    def build_prompt(self, samples, *, store_count: int = 0, own_ids: tuple = (),
                     **_) -> str:
        facts = self.snapshot.lens_facts(self.lens)
        own_hint = (f"You have already deposited {len(own_ids)} claim(s); make a "
                    f"different point.\n" if own_ids else "")
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"You are a {self.lens} analyst. You see ONLY these facts (other "
            f"analysts see different facts):\n\n{facts}\n\n"
            f"{own_hint}"
            f"State ONE specific claim about {self.snapshot.ticker} grounded in a "
            f"number above. Separate the fact from your inference, e.g. "
            f"'P/E is 34 -> modestly rich vs history'. One or two sentences. "
            f"Always cite the number.\n\nCLAIM:"
        )

    async def run(self, store: SignalStore, iterations: int) -> AgentRunStats:
        stats = AgentRunStats(context_record=AgentContextRecord(
            agent_id=self.agent_id, role=self.ROLE))
        own_ids: list[str] = []
        consecutive_dups = 0
        for _ in range(iterations):
            if stats.deposits >= 3:
                break
            stats.iterations += 1
            prompt = self.build_prompt([], own_ids=tuple(own_ids))
            self._assert_no_leak(prompt, [])
            raw = await self.llm.generate(prompt, role=self.ROLE,
                                          max_tokens=self.MAX_TOKENS,
                                          temperature=self.TEMPERATURE)
            content = strip_reasoning((raw or "").strip())
            # Domain junk gate (D3): a stock claim must reference a number.
            if not content or not _has_number(content):
                consecutive_dups += 1
                stats.rejected_dup += 1
                if consecutive_dups >= 3:
                    break
                continue
            sid = store.deposit(
                signal_type=INITIAL, content=content,
                strength=self.DEFAULT_DEPOSIT_STRENGTH, depositor=self.ROLE,
                metadata={"depositor_agent_id": self.agent_id,
                          "partition_id": self.lens,
                          "lens": self.lens,
                          "ticker": self.snapshot.ticker},
            )
            if sid is None:
                consecutive_dups += 1
                stats.rejected_dup += 1
                if consecutive_dups >= 3:
                    break
            else:
                consecutive_dups = 0
                stats.deposits += 1
                own_ids.append(sid)
        return stats


# ---------------------------------------------------------------------------
# DataValidator — numeric VERIFICATION against ground truth (REAL, the D4 fix)
# ---------------------------------------------------------------------------

class DataValidator(BaseAgent):
    """Verifies a signal's numeric claim against the snapshot.

    Deposits VERIFICATION with strength = stock_verify closeness, and stores the
    resolved fact in metadata['atoms'] so projection._build_atoms() picks it up.
    This is what makes verification_score meaningful (vs the ~0 seen in the
    debate run). No LLM call — verification is deterministic arithmetic.
    """
    ROLE = "validator"
    OUTPUT_TYPE = VERIFICATION
    INPUT_TYPE = INITIAL
    MAX_TOKENS = MAX_TOKENS_VALIDATOR
    TEMPERATURE = 0.0
    DEFAULT_DEPOSIT_STRENGTH = 0.5
    MAX_DEPOSITS_PER_ROUND = 4

    def __init__(self, agent_id: str, llm, snapshot: Snapshot, task_prompt: str,
                 strategy: Optional[SamplingStrategy] = None,
                 strategy_name: str = "", **_):
        super().__init__(agent_id, llm)
        self.snapshot = snapshot
        self.task_prompt = task_prompt
        self.strategy = strategy
        self.strategy_name = strategy_name
        self._verified_ids: set[str] = set()

    def _candidates(self, store: SignalStore) -> list[Signal]:
        sigs = store.by_type(INITIAL) + store.by_type(SUPPORT)
        # Verify the strongest not-yet-verified signals that carry a number.
        fresh = [s for s in sigs
                 if s.id not in self._verified_ids and _has_number(s.content)]
        return sorted(fresh, key=lambda s: s.strength, reverse=True)

    def sample(self, store: SignalStore) -> list[Signal]:
        cands = self._candidates(store)
        return cands[:1]

    async def run(self, store: SignalStore, iterations: int) -> AgentRunStats:
        stats = AgentRunStats(context_record=AgentContextRecord(
            agent_id=self.agent_id, role=self.ROLE))
        for _ in range(iterations):
            if stats.deposits >= self.MAX_DEPOSITS_PER_ROUND:
                break
            stats.iterations += 1
            samples = self.sample(store)
            stats.context_record.add_signals([s.id for s in samples])
            if not samples:
                break
            target = samples[0]
            self._verified_ids.add(target.id)
            result = stock_verify.verify_claim(target.content, self.snapshot)
            sid = store.deposit(
                signal_type=VERIFICATION,
                content=result.note,
                strength=result.strength,
                depositor=self.ROLE,
                parent_id=target.id,
                metadata={"depositor_agent_id": self.agent_id,
                          "metric": result.metric,
                          "claimed": result.claimed,
                          "actual": result.actual,
                          "score": round(result.strength, 4),
                          # genome/atom pipeline reads this key:
                          "atoms": [result.as_atom()] if result.metric else []},
            )
            if sid is not None:
                stats.deposits += 1
        return stats


# ---------------------------------------------------------------------------
# ValuationCritic — endorsement/challenge from numeric consistency (REAL)
# ---------------------------------------------------------------------------

class ValuationCritic(BaseAgent):
    """Deposits CRITIQUE_POSITIVE when a claim's number matches ground truth,
    CRITIQUE_NEGATIVE when it contradicts it.

    Gives clusters a positive credibility channel (the D8 fix:
    critic_endorsements were 0 in the debate run because critics only ever went
    negative). Only speaks on numerically checkable claims; stays silent on
    purely qualitative ones (the DataValidator/Hater cover those).
    """
    ROLE = "critic"
    OUTPUT_TYPE = CRITIQUE_NEGATIVE
    INPUT_TYPE = SUPPORT
    MAX_TOKENS = MAX_TOKENS_CRITIC
    TEMPERATURE = 0.0
    DEFAULT_DEPOSIT_STRENGTH = 0.5
    MAX_DEPOSITS_PER_ROUND = 4
    _ENDORSE_THRESHOLD = 0.5

    def __init__(self, agent_id: str, llm, snapshot: Snapshot, task_prompt: str,
                 strategy: Optional[SamplingStrategy] = None,
                 strategy_name: str = "", **_):
        super().__init__(agent_id, llm)
        self.snapshot = snapshot
        self.task_prompt = task_prompt
        self.strategy = strategy
        self.strategy_name = strategy_name
        self._seen: set[str] = set()

    def sample(self, store: SignalStore) -> list[Signal]:
        sigs = store.by_type(SUPPORT) + store.by_type(INITIAL)
        fresh = [s for s in sigs if s.id not in self._seen and _has_number(s.content)]
        return sorted(fresh, key=lambda s: s.strength, reverse=True)[:1]

    async def run(self, store: SignalStore, iterations: int) -> AgentRunStats:
        stats = AgentRunStats(context_record=AgentContextRecord(
            agent_id=self.agent_id, role=self.ROLE))
        for _ in range(iterations):
            if stats.deposits >= self.MAX_DEPOSITS_PER_ROUND:
                break
            stats.iterations += 1
            samples = self.sample(store)
            stats.context_record.add_signals([s.id for s in samples])
            if not samples:
                break
            target = samples[0]
            self._seen.add(target.id)
            result = stock_verify.verify_claim(target.content, self.snapshot)
            # Skip claims we can't resolve numerically — silence, not noise.
            if result.metric is None or result.actual is None:
                continue
            # TODO(refine): also check internal arithmetic (P/E ?= price/EPS).
            if result.strength >= self._ENDORSE_THRESHOLD:
                dep_type, strength = CRITIQUE_POSITIVE, result.strength
                text = f"Endorse: {result.note}"
            else:
                dep_type, strength = CRITIQUE_NEGATIVE, round(1.0 - result.strength, 4)
                text = f"Inconsistent: {result.note}"
            sid = store.deposit(
                signal_type=dep_type, content=text, strength=strength,
                depositor=self.ROLE, parent_id=target.id,
                metadata={"depositor_agent_id": self.agent_id,
                          "metric": result.metric,
                          "score": round(result.strength, 4)},
            )
            if sid is not None:
                stats.deposits += 1
        return stats


# ---------------------------------------------------------------------------
# ThesisDeveloper — develops INITIALs into bull/bear SUPPORT (thin subclass)
# ---------------------------------------------------------------------------

class ThesisDeveloper(Developer):
    """Develops a claim with supporting evidence, anticipating dissent.

    Inherits Developer.sample()/run() (gap-fill + dissent stashing). Overrides
    the prompt for stock framing.

    TODO(stage-3): ground development in snapshot facts and respect as_of (the
    inherited Developer.sample() may issue an agentic web search whose results
    are NOT point-in-time — disable or date-clamp it for historical backtests).
    """
    ROLE = "developer"
    MAX_TOKENS = MAX_TOKENS_FORAGER
    TEMPERATURE = 0.6
    DEFAULT_DEPOSIT_STRENGTH = 0.6

    async def run(self, store: SignalStore, iterations: int) -> AgentRunStats:
        # Nothing to develop if no INITIAL exists (e.g. a silent field where the
        # number-gate rejected every scout claim). Depositing SUPPORT with no
        # parent would orphan it and trip the partition-leak guard, so no-op.
        if not store.by_type(INITIAL):
            return AgentRunStats(context_record=AgentContextRecord(
                agent_id=self.agent_id, role=self.ROLE))
        return await super().run(store, iterations)

    def sample(self, store: SignalStore) -> list[Signal]:
        """Like Developer.sample (gap-fill + strategy + dissent stash) but WITHOUT
        the inherited live web search — that is not point-in-time safe for a
        historical backtest (and would hit the network in tests). Evidence comes
        from the snapshot via the prompt, not ad-hoc retrieval."""
        import random
        _MIN_SUPPORT = 2
        underserved = store.signals_with_few_children_of_type(INITIAL, SUPPORT, _MIN_SUPPORT)
        if underserved:
            weights = [max(0.01, s.strength) for s in underserved]
            target = random.choices(underserved, weights=weights, k=1)
        else:
            target = self.strategy(store, self.INPUT_TYPE, 1)
            if not target:
                target = store.sample_weighted(self.INPUT_TYPE, 1)
        self._stashed_dissent = None
        self._stashed_retrieval = []
        self._stashed_query = ""
        if target:
            children = [store.get(cid) for cid in store.by_parent(target[0].id)]
            children = [c for c in children if c is not None]
            dissent = [c for c in children if c.type in (CRITIQUE_NEGATIVE, OBJECTION)]
            if dissent:
                self._stashed_dissent = max(dissent, key=lambda c: c.strength)
        return target

    def build_prompt(self, samples, *, store_count: int = 0, own_ids: tuple = ()) -> str:
        if not samples:
            return (f"TASK: {self.task_prompt}\n\nNo claims yet. State one "
                    f"number-grounded observation.\n\nDEVELOPMENT:")
        s = samples[0]
        dissent = ""
        if getattr(self, "_stashed_dissent", None) is not None:
            d = self._stashed_dissent
            dissent = f"\nA challenge was raised — address it: [{d.id}]: {d.content}\n"
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"Develop this claim with ONE piece of supporting evidence (bull or "
            f"bear), citing a specific number. Do not restate it.\n\n"
            f"---SIGNAL [{s.id}]---\n{s.content}\n---END SIGNAL---\n{dissent}\n"
            f"{type_parent_instruction()}\nDEVELOPMENT:"
        )


# ---------------------------------------------------------------------------
# RiskHater — cycles a risk checklist (REAL prompt, mirrors EdgeCaseHater)
# ---------------------------------------------------------------------------

_RISK_CHECKLIST = [
    "valuation risk (multiple compression if growth slows)",
    "growth deceleration / decelerating revenue or EPS",
    "margin compression (cost inflation, pricing pressure)",
    "competitive threat / market-share loss",
    "balance-sheet / liquidity risk (debt, refinancing)",
    "customer or segment concentration",
    "regulatory, legal, or litigation exposure",
    "macro / rate sensitivity / cyclicality",
    "sentiment reversal / crowded positioning",
]


class RiskHater(BaseAgent):
    """Names a specific downside risk the bull cluster ignores.

    Cycles a canonical risk checklist (like coding's EdgeCaseHater cycles edge
    cases) so adversarial pressure is systematic, not whatever the model
    fixates on.
    """
    ROLE = "hater"
    OUTPUT_TYPE = OBJECTION
    INPUT_TYPE = INITIAL
    MAX_TOKENS = MAX_TOKENS_HATER
    TEMPERATURE = 0.8
    DEFAULT_DEPOSIT_STRENGTH = 0.6
    MAX_DEPOSITS_PER_ROUND = 2

    def __init__(self, agent_id: str, llm, task_prompt: str, snapshot: Optional[Snapshot] = None, **_):
        super().__init__(agent_id, llm)
        self.task_prompt = task_prompt
        self.snapshot = snapshot
        self._idx = 0

    async def run(self, store: SignalStore, iterations: int) -> AgentRunStats:
        # No thesis to challenge -> no-op (avoids emitting objections into the void).
        if not (store.by_type(INITIAL) or store.by_type(SUPPORT)):
            return AgentRunStats(context_record=AgentContextRecord(
                agent_id=self.agent_id, role=self.ROLE))
        return await super().run(store, iterations)

    def sample(self, store: SignalStore) -> list[Signal]:
        sigs = store.by_type(SUPPORT) + store.by_type(INITIAL)
        if not sigs:
            return []
        return [max(sigs, key=lambda s: s.strength)]

    def build_prompt(self, samples, *, store_count: int = 0, own_ids: tuple = ()) -> str:
        risk = _RISK_CHECKLIST[self._idx % len(_RISK_CHECKLIST)]
        self._idx += 1
        if not samples:
            return (f"TASK: {self.task_prompt}\n\nNo thesis yet. Skip.\n\nOBJECTION: (none)")
        s = samples[0]
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"The forming thesis is:\n  [{s.id}]: {s.content}\n\n"
            f"Does it underweight this risk: {risk}? If so, name the specific "
            f"downside in one sentence with a number if possible; else say "
            f"'unlikely'.\n\n{type_parent_instruction()}\nOBJECTION:"
        )

    def parent_id_for_deposit(self, samples):
        return max(samples, key=lambda s: s.strength).id if samples else None


# ---------------------------------------------------------------------------
# EquityBriefSynthesizer — report + the gradable PREDICTION block
# ---------------------------------------------------------------------------

# First-cut mapping from net field-stance to a horizon return estimate. This is
# a TRANSPARENT placeholder: tune MAX_EXPECTED_MOVE_PCT and the mapping against
# the historical DB (Stage 5) before trusting the magnitude. The SIGN/direction
# is the robust part; the magnitude is the part to calibrate.
MAX_EXPECTED_MOVE_PCT = 10.0
_LONG_NET_THRESHOLD = 0.15
_SHORT_NET_THRESHOLD = -0.40   # short disabled by default (POC is long/avoid)
_ENABLE_SHORT = False

_BULL_LEX = ("undervalued", "cheap", "growth", "beat", "expanding", "upside",
             "strong", "accelerat", "raised", "outperform", "tailwind", "buy")
_BEAR_LEX = ("overvalued", "rich", "expensive", "decelerat", "miss", "compress",
             "weak", "downside", "headwind", "risk", "litigation", "sell", "decline")


def _stance_weight(text: str) -> float:
    """+1 bullish .. -1 bearish lexical lean of a cluster representative.

    TODO(refine): replace lexicon with an embedding/LLM stance read; the
    lexicon is a deterministic, MOCK-safe first cut."""
    low = (text or "").lower()
    bull = sum(low.count(w) for w in _BULL_LEX)
    bear = sum(low.count(w) for w in _BEAR_LEX)
    if bull == bear == 0:
        return 0.0
    return (bull - bear) / (bull + bear)


class EquityBriefSynthesizer(Synthesizer):
    """Renders the equity brief and the gradable prediction.json.

    Deterministic (LLM-free) so it is MOCK-safe and testable. Prose enrichment
    via the parent Synthesizer's per-cluster rendering is a TODO(stage-3); the
    structured PREDICTION block and metrics table are real now because they are
    what the backtest grades.

    The snapshot is read from self._snapshot (set by run_swarm wiring); horizon
    from self._horizon_days (default 21 ≈ 1 month).
    """

    async def synthesize(self, store, has_validators=True, prior_rejections=None,
                         prior_consensus=None, output_dir=None, task_type=None):
        from core.projection import build_projection
        from agents.synthesizer import _build_citations, _build_lineage_dot

        projection = build_projection(
            store, has_validators=has_validators,
            prior_rejections=prior_rejections, prior_consensus=prior_consensus,
            task_type=task_type)

        snapshot: Optional[Snapshot] = getattr(self, "_snapshot", None)
        horizon = int(getattr(self, "_horizon_days", 21))
        ticker = snapshot.ticker if snapshot else getattr(self, "_ticker", "UNKNOWN")

        pred = self.build_prediction(projection, store, snapshot, ticker, horizon)
        self._last_prediction = pred   # so in-process callers can read it sans file
        answer = self._render_report(projection, store, snapshot, pred)

        if output_dir is not None:
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                (Path(output_dir) / "prediction.json").write_text(
                    json.dumps(pred, indent=2), encoding="utf-8")
            except Exception as exc:  # never let IO crash synthesis
                print(f"[stock-synth] could not write prediction.json: {exc}")

        citations = _build_citations(projection, store)
        lineage_dot = _build_lineage_dot(projection, store)
        return answer, citations, lineage_dot

    # ---- the gradable artifact (REAL) -----------------------------------
    def build_prediction(self, projection, store, snapshot, ticker, horizon) -> dict:
        """Aggregate surviving clusters into {direction, predicted_return_pct,
        confidence}. Sign is robust; magnitude is the calibration target."""
        surviving = list(getattr(projection, "surviving", []))
        bull_w = bear_w = 0.0
        verified = 0
        for cp in surviving:
            rep = store.get(cp.representative_id)
            if rep is None:
                continue
            stance = _stance_weight(rep.content)
            # weight by how well-supported and verified the cluster is
            w = max(0.0, getattr(cp, "support_diversity", 1)) * \
                (1.0 + max(0.0, getattr(cp, "verification_score", 0.0)))
            if stance >= 0:
                bull_w += stance * w
            else:
                bear_w += (-stance) * w
            if getattr(cp, "verification_score", 0.0) > 0:
                verified += 1

        total = bull_w + bear_w
        net = (bull_w - bear_w) / total if total > 1e-9 else 0.0
        if net >= _LONG_NET_THRESHOLD:
            direction = "long"
        elif _ENABLE_SHORT and net <= _SHORT_NET_THRESHOLD:
            direction = "short"
        else:
            direction = "avoid"
        predicted_return_pct = round(net * MAX_EXPECTED_MOVE_PCT, 2)
        # confidence: verification coverage × decisiveness of the net lean
        cov = (verified / len(surviving)) if surviving else 0.0
        confidence = round(min(1.0, 0.25 + 0.5 * cov + 0.25 * abs(net)), 3)

        return {
            "schema": "stock_prediction_v1",
            "ticker": ticker,
            "as_of_date": snapshot.as_of.isoformat() if snapshot else None,
            "horizon_days": horizon,
            "direction": direction,
            "predicted_return_pct": predicted_return_pct,
            "confidence": confidence,
            "n_surviving_clusters": len(surviving),
            "verified_clusters": verified,
            "key_claim_ids": [cp.representative_id for cp in surviving[:6]],
            "disclaimer": "Not financial advice; backtest artifact.",
        }

    # ---- human report (deterministic; prose enrichment is TODO) ---------
    def _render_report(self, projection, store, snapshot, pred) -> str:
        L = []
        L.append(f"# {pred['ticker']} — Equity Brief (as of {pred['as_of_date']})")
        L.append("")
        L.append(f"## VERDICT: {pred['direction'].upper()} "
                 f"({pred['predicted_return_pct']:+.1f}% over {pred['horizon_days']} "
                 f"trading days, confidence {pred['confidence']:.0%})")
        L.append("")
        if snapshot is not None:
            L.append("## Key metrics")
            for m in snapshot.present_metrics():
                from core.stock_data import _fmt
                L.append(f"- {m}: {_fmt(m, snapshot.get(m))}")
            L.append("")
        vf = _verified_facts_lines(store)
        if vf:
            L.append("## Verified facts (swarm claim vs ground truth)")
            L += vf
            L.append("")
        surviving = list(getattr(projection, "surviving", []))
        bull = [cp for cp in surviving
                if _stance_weight(_rep_text(store, cp)) > 0]
        bear = [cp for cp in surviving
                if _stance_weight(_rep_text(store, cp)) < 0]
        L.append("## Bull case")
        L += [f"- {_rep_text(store, cp)} [{cp.representative_id}]" for cp in bull[:5]] or ["- (none surfaced)"]
        L.append("")
        L.append("## Bear case")
        L += [f"- {_rep_text(store, cp)} [{cp.representative_id}]" for cp in bear[:5]] or ["- (none surfaced)"]
        L.append("")
        L.append("## Key risks")
        objs = store.by_type(OBJECTION)
        L += [f"- {o.content} [{o.id}]" for o in sorted(objs, key=lambda s: s.strength, reverse=True)[:5]] or ["- (none surfaced)"]
        L.append("")
        _dissent = sorted(objs, key=lambda s: s.strength, reverse=True)
        if _dissent:
            L.append("## What would change the view")
            L.append(f"- Watch the strongest challenge: {_dissent[0].content}")
            L.append("")
        # LLM prose synthesis per cluster (parent Synthesizer render methods) is
        # an optional enrichment; the deterministic block above is the fallback.
        L.append("> Not financial advice; generated for backtest evaluation.")
        return "\n".join(L)


def _rep_text(store, cp) -> str:
    rep = store.get(cp.representative_id)
    return rep.content if rep else ""


def _verified_facts_lines(store, max_facts: int = 8) -> list:
    """Surface DataValidator evidence: each metric's claimed-vs-actual value,
    deduplicated by metric (keep the best-scoring verification). This makes the
    swarm's grounding visible instead of hidden in VERIFICATION metadata."""
    best: dict = {}
    for v in store.by_type(VERIFICATION):
        m = v.metadata.get("metric")
        actual = v.metadata.get("actual")
        claimed = v.metadata.get("claimed")
        if not m or actual is None or claimed is None:
            continue
        score = float(v.metadata.get("score", v.strength) or 0.0)
        if m not in best or score > best[m][0]:
            best[m] = (score, claimed, actual)
    lines = []
    for m, (score, claimed, actual) in sorted(
            best.items(), key=lambda kv: kv[1][0], reverse=True)[:max_facts]:
        mark = "verified" if score >= 0.5 else "off"
        lines.append(f"- {m}: claimed {claimed:g} vs actual {actual:g} "
                     f"— {mark} ({score:.0%})")
    return lines


# ---------------------------------------------------------------------------
# Construction helper — encapsulates the fiddly per-role wiring so run_swarm's
# stock branch stays a thin call. (The stock roles' constructors differ from
# the default agents: scouts/validators/critics/haters need the snapshot.)
# ---------------------------------------------------------------------------

def build_stock_agents(
    llm_for,
    snapshot: Snapshot,
    task_prompt: str,
    *,
    horizon_days: int = 21,
    num_developers: int = 2,
    num_critics: int = 1,
    num_haters: int = 1,
    num_validators: int = 1,
) -> dict:
    """Build all stock agents ready to run, grouped by role.

    `llm_for(role: str) -> llm` returns the engine for a role — pass
    `router.engine_for` for the Groq path, or `lambda _r: llm` for a single
    model. Returns {"scout": [...], "developer": [...], "critic": [...],
    "hater": [...], "validator": [...], "synthesizer": <agent>}.

    One LensScout per analytical lens (the partition set). The synthesizer
    carries the snapshot + horizon so it can render the metrics table and the
    gradable prediction.json.
    """
    from core.sampling import under_supported_clusters  # default dev strategy

    scouts = [
        LensScout(f"scout_{lens}", llm_for("scout"), snapshot, lens, task_prompt)
        for lens in LENSES
    ]
    developers = [
        ThesisDeveloper(f"dev_{i}", llm_for("developer"),
                        under_supported_clusters, "under_supported", task_prompt)
        for i in range(num_developers)
    ]
    critics = [
        ValuationCritic(f"critic_{i}", llm_for("critic"), snapshot, task_prompt)
        for i in range(num_critics)
    ]
    haters = [
        RiskHater(f"hater_{i}", llm_for("hater"), task_prompt, snapshot=snapshot)
        for i in range(num_haters)
    ]
    validators = [
        DataValidator(f"validator_{i}", llm_for("validator"), snapshot, task_prompt)
        for i in range(num_validators)
    ]
    # Synthesizer.__init__(self, llm, task_prompt) — see agents/synthesizer.py.
    synth = EquityBriefSynthesizer(llm_for("synthesizer"), task_prompt)
    # The synthesizer reads these for rendering + the prediction artifact.
    synth._snapshot = snapshot
    synth._ticker = snapshot.ticker
    synth._horizon_days = horizon_days

    return {"scout": scouts, "developer": developers, "critic": critics,
            "hater": haters, "validator": validators, "synthesizer": synth}
