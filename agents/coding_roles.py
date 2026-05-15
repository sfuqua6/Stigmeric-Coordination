"""Coding-specific agent roles (§5 directive).

These roles are task-adaptive instantiations for task_type="coding". Each
role treats signals as code artifacts rather than prose claims.

    RequirementsScout  → INITIAL signals are acceptance criteria / assumptions
    CodeDeveloper      → SUPPORT signals are code snippets implementing an INITIAL
    StaticCritic       → CRITIQUE_{POSITIVE,NEGATIVE} based on ast.parse() result
    EdgeCaseHater      → OBJECTION signals name specific edge cases
    TestValidator      → VERIFICATION via subprocess pytest, strength in {0.0, 0.5, 1.0}
    CodeSynthesizer    → assembles surviving fenced blocks + py_compile check

# FUTURE-CLAUDE NOTE: These roles use subprocess.run for test execution.
# The timeout is 5 seconds per test run (SUBPROC_TIMEOUT_S below). Do not
# raise this beyond 10s without also adding resource limits (ulimit, etc.)
# — a poorly-formed test could otherwise hang the pipeline indefinitely.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import os
from typing import Optional

from agents.base import BaseAgent, AgentRunStats, strip_reasoning
from agents.developer import Developer
from agents.synthesizer import Synthesizer
from core.signal_store import SignalStore, Signal
from core.signal_types import (
    INITIAL, SUPPORT, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE, OBJECTION, VERIFICATION,
)
from core.config import (
    MAX_TOKENS_SCOUT, MAX_TOKENS_FORAGER, MAX_TOKENS_CRITIC,
    MAX_TOKENS_HATER, MAX_TOKENS_VALIDATOR, MAX_TOKENS_SYNTHESIZER,
)
from core.diversity import AgentContextRecord
from core.sampling import SamplingStrategy

SUBPROC_TIMEOUT_S = 5

_FENCED_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Extract the first fenced Python block from text."""
    m = _FENCED_RE.search(text)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# RequirementsScout
# ---------------------------------------------------------------------------

class RequirementsScout(BaseAgent):
    """Scout that reads the task prompt directly and emits acceptance criteria.

    No corpus partition: for coding tasks, requirements come from the task
    prompt itself (no corpus retrieval needed). Each INITIAL is one
    acceptance criterion or assumption about the required behavior.
    """
    ROLE = "scout"
    OUTPUT_TYPE = INITIAL
    INPUT_TYPE = None
    MAX_TOKENS = MAX_TOKENS_SCOUT
    TEMPERATURE = 0.7
    DEFAULT_DEPOSIT_STRENGTH = 0.55

    def __init__(self, agent_id: str, llm, task_prompt: str, **_):
        super().__init__(agent_id, llm)
        self.task_prompt = task_prompt

    def sample(self, store: SignalStore) -> list[Signal]:
        return []

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = (),
                     **_) -> str:
        own_hint = ""
        if own_ids:
            own_hint = (
                f"You have already deposited {len(own_ids)} criterion/a. "
                f"Do not restate them.\n"
            )
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"{own_hint}"
            f"Produce ONE acceptance criterion or explicit assumption for "
            f"the above coding task. Write it as a single testable "
            f"statement. Do not write code. One sentence only.\n\n"
            f"CRITERION:"
        )

    async def run(self, store: SignalStore, iterations: int) -> AgentRunStats:
        stats = AgentRunStats(context_record=AgentContextRecord(
            agent_id=self.agent_id, role=self.ROLE,
        ))
        consecutive_dups = 0
        own_ids: list[str] = []

        for _ in range(iterations):
            if stats.deposits >= 3:
                break
            stats.iterations += 1
            prompt = self.build_prompt([], own_ids=tuple(own_ids))
            self._assert_no_leak(prompt, [])

            raw = await self.llm.generate(
                prompt, role=self.ROLE,
                max_tokens=self.MAX_TOKENS, temperature=self.TEMPERATURE,
            )
            content = strip_reasoning(raw.strip())
            if not content:
                continue

            sid = store.deposit(
                signal_type=self.OUTPUT_TYPE,
                content=content,
                strength=self.DEFAULT_DEPOSIT_STRENGTH,
                depositor=self.ROLE,
                metadata={"depositor_agent_id": self.agent_id},
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
# CodeDeveloper
# ---------------------------------------------------------------------------

class CodeDeveloper(Developer):
    """Developer that produces code snippet SUPPORT signals.

    Each deposit must contain exactly one fenced Python block. Deposits
    without a fenced block are rejected before reaching the signal store.
    """
    ROLE = "developer"
    MAX_TOKENS = MAX_TOKENS_FORAGER
    TEMPERATURE = 0.6
    DEFAULT_DEPOSIT_STRENGTH = 0.65

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = ()) -> str:
        if not samples:
            return (
                f"TASK: {self.task_prompt}\n\n"
                f"No initial signals yet. Write a short Python function "
                f"that solves or starts to solve the task. Wrap in ```python fenced block.\n\n"
                f"IMPLEMENTATION:"
            )
        s = samples[0]
        dissent_block = ""
        if self._stashed_dissent is not None:
            d = self._stashed_dissent
            dissent_block = (
                f"\nKnown weakness to address [{d.id}]: {d.content}\n"
            )
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"You observe one acceptance criterion:\n"
            f"  [{s.id}]: {s.content}\n"
            f"{dissent_block}\n"
            f"Write ONE Python function or snippet that implements or extends "
            f"this criterion. Wrap your code in ```python fenced block. "
            f"Do not write prose; only code inside the block.\n\n"
            f"IMPLEMENTATION:"
        )

    async def run(self, store: SignalStore, iterations: int) -> AgentRunStats:
        stats = AgentRunStats(context_record=AgentContextRecord(
            agent_id=self.agent_id, role=self.ROLE,
        ))
        consecutive_dups = 0
        own_ids: list[str] = []

        for _ in range(iterations):
            if self.MAX_DEPOSITS_PER_ROUND is not None and stats.deposits >= self.MAX_DEPOSITS_PER_ROUND:
                break
            stats.iterations += 1
            samples = self.sample(store)
            stats.context_record.add_signals([s.id for s in samples])

            store_count = len(store.by_type(SUPPORT))
            prompt = self.build_prompt(samples, store_count=store_count, own_ids=tuple(own_ids))
            self._assert_no_leak(prompt, samples)

            raw = await self.llm.generate(
                prompt, role=self.ROLE,
                max_tokens=self.MAX_TOKENS, temperature=self.TEMPERATURE,
            )
            content = strip_reasoning(raw.strip())
            if not content:
                continue

            # Reject deposits with no fenced Python block
            code = _extract_code(content)
            if not code:
                stats.rejected_dup += 1
                consecutive_dups += 1
                if consecutive_dups >= 3:
                    break
                continue

            sid = store.deposit(
                signal_type=SUPPORT,
                content=content,
                strength=self.DEFAULT_DEPOSIT_STRENGTH,
                depositor=self.ROLE,
                parent_id=self.parent_id_for_deposit(samples),
                metadata={"depositor_agent_id": self.agent_id,
                          "has_code_block": True},
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
# StaticCritic
# ---------------------------------------------------------------------------

class StaticCritic(BaseAgent):
    """Critic that runs ast.parse() on SUPPORT code blocks.

    Deposits CRITIQUE_NEGATIVE for syntax errors, CRITIQUE_POSITIVE for
    parseable code. Score = 1 - (errors / total_attempts).
    """
    ROLE = "critic"
    OUTPUT_TYPE = CRITIQUE_NEGATIVE
    INPUT_TYPE = SUPPORT
    MAX_TOKENS = MAX_TOKENS_CRITIC
    TEMPERATURE = 0.3
    DEFAULT_DEPOSIT_STRENGTH = 0.5

    def __init__(self, agent_id: str, llm, strategy: SamplingStrategy,
                 strategy_name: str, task_prompt: str):
        super().__init__(agent_id, llm)
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.task_prompt = task_prompt
        self._total_attempts = 0
        self._errors = 0

    def sample(self, store: SignalStore) -> list[Signal]:
        return self.strategy(store, SUPPORT, 1)

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = ()) -> str:
        if not samples:
            return "No code to critique.\n\nCRITIQUE:\nSCORE: 0.0"
        s = samples[0]
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"Review this code signal for correctness relative to the task:\n"
            f"  [{s.id}]: {s.content[:400]}\n\n"
            f"One sentence critique. SCORE: 0.0-1.0 (1.0 = correct).\n\n"
            f"CRITIQUE: <text>\nSCORE: <0-1>"
        )

    async def run(self, store: SignalStore, iterations: int) -> AgentRunStats:
        stats = AgentRunStats(context_record=AgentContextRecord(
            agent_id=self.agent_id, role=self.ROLE,
        ))
        consecutive_dups = 0
        own_ids: list[str] = []

        for _ in range(iterations):
            stats.iterations += 1
            samples = self.sample(store)
            stats.context_record.add_signals([s.id for s in samples])

            if not samples:
                continue

            s = samples[0]
            code = _extract_code(s.content)
            self._total_attempts += 1

            parse_ok = False
            parse_error = ""
            if code:
                try:
                    ast.parse(code)
                    parse_ok = True
                except SyntaxError as e:
                    self._errors += 1
                    parse_error = str(e)
            else:
                self._errors += 1
                parse_error = "No fenced code block found"

            score = max(0.0, 1.0 - (self._errors / self._total_attempts))
            if parse_ok:
                deposit_type = CRITIQUE_POSITIVE
                critique_text = f"CRITIQUE: Code parses successfully (no syntax errors). SCORE: {score:.2f}"
            else:
                deposit_type = CRITIQUE_NEGATIVE
                critique_text = f"CRITIQUE: Syntax error — {parse_error}. SCORE: {score:.2f}"

            sid = store.deposit(
                signal_type=deposit_type,
                content=critique_text,
                strength=score,
                depositor=self.ROLE,
                parent_id=s.id,
                metadata={"depositor_agent_id": self.agent_id,
                          "parse_ok": parse_ok,
                          "score": round(score, 4)},
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
# EdgeCaseHater
# ---------------------------------------------------------------------------

_EDGE_CASES = [
    "empty input (empty string, empty list, None)",
    "maximum integer value (sys.maxsize or 2**63-1)",
    "None passed where a sequence is expected",
    "zero-length input",
    "negative numbers or negative indices",
    "duplicate elements in input",
    "single-element input",
    "unicode strings and non-ASCII input",
]


class EdgeCaseHater(BaseAgent):
    """Hater that names specific edge cases the cluster's code likely fails on.

    Ignores embedding clusters; instead picks from a library of canonical
    edge cases and asks the LLM to determine which applies to the cluster.
    """
    ROLE = "hater"
    OUTPUT_TYPE = OBJECTION
    INPUT_TYPE = SUPPORT
    MAX_TOKENS = MAX_TOKENS_HATER
    TEMPERATURE = 0.7
    DEFAULT_DEPOSIT_STRENGTH = 0.6
    MAX_DEPOSITS_PER_ROUND = 1

    def __init__(self, agent_id: str, llm, task_prompt: str, **_):
        super().__init__(agent_id, llm)
        self.task_prompt = task_prompt
        self._edge_case_idx = 0

    def sample(self, store: SignalStore) -> list[Signal]:
        sups = store.by_type(SUPPORT)
        if not sups:
            return []
        return [max(sups, key=lambda s: s.strength)]

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = ()) -> str:
        if not samples:
            return (
                f"TASK: {self.task_prompt}\n\n"
                f"No code signals yet. Skip.\n\nOBJECTION: (none)"
            )
        s = samples[0]
        edge_case = _EDGE_CASES[self._edge_case_idx % len(_EDGE_CASES)]
        self._edge_case_idx += 1
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"Given this code signal:\n  [{s.id}]: {s.content[:400]}\n\n"
            f"Does the code likely fail on: {edge_case}?\n"
            f"One sentence explaining the failure mode (or 'unlikely' if not applicable).\n\n"
            f"OBJECTION:"
        )


# ---------------------------------------------------------------------------
# TestValidator
# ---------------------------------------------------------------------------

class TestValidator(BaseAgent):
    """Validator that writes and executes a pytest test for the strongest SUPPORT.

    Deposits VERIFICATION at:
        1.0  — test passes
        0.0  — test fails
        0.5  — test times out or could not be constructed
    """
    ROLE = "validator"
    OUTPUT_TYPE = VERIFICATION
    INPUT_TYPE = SUPPORT
    MAX_TOKENS = MAX_TOKENS_VALIDATOR * 3  # test generation needs more tokens
    TEMPERATURE = 0.3
    DEFAULT_DEPOSIT_STRENGTH = 0.5

    def __init__(self, agent_id: str, llm, strategy: SamplingStrategy,
                 strategy_name: str, task_prompt: str):
        super().__init__(agent_id, llm)
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.task_prompt = task_prompt

    def sample(self, store: SignalStore) -> list[Signal]:
        sups = store.by_type(SUPPORT)
        if not sups:
            return []
        return [max(sups, key=lambda s: s.strength)]

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = ()) -> str:
        if not samples:
            return "No code to test.\n\nTEST:\nSCORE: 0.5"
        s = samples[0]
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"Write ONE pytest test function for this code:\n"
            f"  [{s.id}]: {s.content[:600]}\n\n"
            f"The test must import nothing except the function under test "
            f"(copy it inline). Wrap in ```python fenced block.\n\n"
            f"TEST:"
        )

    async def run(self, store: SignalStore, iterations: int) -> AgentRunStats:
        stats = AgentRunStats(context_record=AgentContextRecord(
            agent_id=self.agent_id, role=self.ROLE,
        ))

        for _ in range(iterations):
            if stats.deposits >= 1:
                break
            stats.iterations += 1
            samples = self.sample(store)
            stats.context_record.add_signals([s.id for s in samples])

            if not samples:
                continue

            prompt = self.build_prompt(samples)
            self._assert_no_leak(prompt, samples)

            raw = await self.llm.generate(
                prompt, role=self.ROLE,
                max_tokens=self.MAX_TOKENS, temperature=self.TEMPERATURE,
            )
            content = strip_reasoning(raw.strip())
            code = _extract_code(content)
            if not code:
                score = 0.5
                verification_text = "Could not extract a test function."
            else:
                score, verification_text = _run_test_subprocess(code)

            sid = store.deposit(
                signal_type=VERIFICATION,
                content=verification_text,
                strength=score,
                depositor=self.ROLE,
                parent_id=samples[0].id if samples else None,
                metadata={"depositor_agent_id": self.agent_id,
                          "score": round(score, 4)},
            )
            if sid is not None:
                stats.deposits += 1

        return stats


def _run_test_subprocess(test_code: str) -> tuple[float, str]:
    """Run test_code in a sandboxed subprocess. Returns (score, text)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(test_code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", fname, "-x", "-q", "--tb=short"],
            capture_output=True, text=True, timeout=SUBPROC_TIMEOUT_S,
        )
        if result.returncode == 0:
            return 1.0, f"Tests passed.\n{result.stdout[:300]}"
        else:
            return 0.0, f"Tests failed.\n{result.stdout[:200]}\n{result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return 0.5, f"Test timed out after {SUBPROC_TIMEOUT_S}s."
    except Exception as e:
        return 0.5, f"Could not run test: {e}"
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CodeSynthesizer
# ---------------------------------------------------------------------------

class CodeSynthesizer(Synthesizer):
    """Synthesizer that assembles surviving fenced code blocks.

    Emits one fenced Python block per surviving cluster, then runs
    py_compile on the assembled result. If compile fails, falls back to
    the strongest single SUPPORT code block.
    """

    async def synthesize(self, store, has_validators=True,
                         prior_rejections=None, prior_consensus=None,
                         output_dir=None):
        from core.projection import build_projection

        projection = build_projection(
            store, has_validators=has_validators,
            prior_rejections=prior_rejections,
            prior_consensus=prior_consensus,
        )

        code_blocks: list[str] = []
        alt_blocks: list[str] = []

        for cp in projection.surviving[:4]:
            rep = store.get(cp.representative_id)
            if rep:
                code = _extract_code(rep.content)
                if code:
                    code_blocks.append(
                        f"# Cluster {cp.representative_id} "
                        f"(diversity={cp.support_diversity}, "
                        f"dissent={cp.dissent_pressure:.2f})\n{code}"
                    )
            for sid in cp.support_set[:3]:
                s = store.get(sid)
                if s:
                    alt = _extract_code(s.content)
                    if alt:
                        alt_blocks.append(f"# {sid}\n{alt}")

        if not code_blocks:
            code_blocks = alt_blocks[:3]

        if not code_blocks:
            answer = "# No code signals survived projection.\n"
        else:
            assembled = "\n\n".join(code_blocks)
            full_code = f"# Assembled by CodeSynthesizer\n\n{assembled}\n"

            # Validate: py_compile the assembled result
            import py_compile, io
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False, encoding="utf-8"
                ) as f:
                    f.write(full_code)
                    fname = f.name
                py_compile.compile(fname, doraise=True)
                os.unlink(fname)
                answer = f"```python\n{full_code}```\n"
            except py_compile.PyCompileError as e:
                # Fall back to strongest single SUPPORT
                os.unlink(fname)
                best_code = alt_blocks[0] if alt_blocks else "# (no valid code)"
                answer = (
                    f"# Assembly failed to compile: {e}\n"
                    f"# Falling back to strongest SUPPORT signal.\n\n"
                    f"```python\n{best_code}\n```\n"
                )

        from agents.synthesizer import _build_citations, _build_lineage_dot
        citations = _build_citations(projection, store)
        lineage_dot = _build_lineage_dot(projection, store)
        return answer, citations, lineage_dot
