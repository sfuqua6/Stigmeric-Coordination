"""Hater — challenges the consensus cluster.

The hater does NOT sample individual signals to attack one at a time.
That would be too local — and it would let the hater anchor on whatever
single signal it happened to draw, which is a stigmergic violation
disguised as adversarial pressure.

Real clustering (§5 directive)
--------------------------------
The old consensus_summary(k=3) returned the top-K signals by strength, not a
genuine semantic cluster. This meant the hater challenged whichever three
signals happened to be strongest, which was not necessarily a coherent cluster.

The hater now uses SignalStore.cluster_signals_dbscan() to find the largest
genuine cluster in embedding space (cosine-DBSCAN with eps=0.35). It receives
representatives of this cluster as its context. The prompt explicitly tells
the model that this is a real cluster in embedding space, not just the top-K
by strength.

If no embeddings are available (embedder disabled), falls back to the old
top-K behaviour with a warning.
"""

from __future__ import annotations

from agents.base import BaseAgent, type_parent_instruction
from core.signal_store import SignalStore, Signal
from core.signal_types import INITIAL, OBJECTION
from core.config import MAX_TOKENS_HATER


class Hater(BaseAgent):
    ROLE = "hater"
    OUTPUT_TYPE = OBJECTION
    INPUT_TYPE = INITIAL
    MAX_TOKENS = MAX_TOKENS_HATER
    TEMPERATURE = 0.85
    DEFAULT_DEPOSIT_STRENGTH = 0.6
    # Cap at 1 OBJECTION per round per hater. Two haters → 2 OBJECTIONs/round max.
    # Without this cap both haters burn all 8 iterations producing paraphrased dissent.
    MAX_DEPOSITS_PER_ROUND = 1

    def __init__(self, agent_id: str, llm, task_prompt: str, target_type: str = INITIAL):
        super().__init__(agent_id, llm)
        self.task_prompt = task_prompt
        self.target_type = target_type
        self._last_rep_ids: list[str] = []
        self._last_cluster_id: str = ""
        self._used_dbscan = False

    def sample(self, store: SignalStore) -> list[Signal]:
        # Fix C': prefer ClusterRegistry over DBSCAN (pre-built at deposit time).
        live_clusters = store.get_live_clusters(self.target_type)
        if live_clusters:
            # Pick the largest live cluster; sample up to 3 strongest members.
            cluster_id, member_ids = max(live_clusters, key=lambda x: len(x[1]))
            self._last_cluster_id = cluster_id
            self._used_dbscan = True
            members = [store.get(mid) for mid in member_ids if store.get(mid)]
            reps = sorted(members, key=lambda s: s.strength, reverse=True)[:3]
        else:
            # Fallback: DBSCAN (no embedder or empty registry)
            self._last_cluster_id = ""
            clusters = store.cluster_signals_dbscan(self.target_type, eps=0.35)
            if clusters and len(clusters[0]) > 0:
                self._used_dbscan = True
                largest = clusters[0]
                reps = sorted(largest, key=lambda s: s.strength, reverse=True)[:3]
            else:
                self._used_dbscan = False
                summary = store.consensus_summary(self.target_type, k=3)
                rep_signals = [store.get(r["id"]) for r in summary["representatives"]]
                reps = [s for s in rep_signals if s is not None]

        self._last_rep_ids = [s.id for s in reps]
        return reps

    def extra_deposit_metadata(self) -> dict:
        meta = {}
        if self._last_cluster_id:
            meta["targets_cluster_id"] = self._last_cluster_id
        return meta

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = ()) -> str:
        if not samples:
            return (
                f"TASK: {self.task_prompt}\n\n"
                f"No consensus has formed yet. Skip.\n\n"
                f"OBJECTION: (none)"
            )
        count_hint = (
            f"There are currently {store_count} objections in the store. "
            f"Produce a structurally distinct adversarial challenge.\n"
        )
        if self._used_dbscan:
            cluster_desc = (
                f"a real semantic cluster in embedding space (not just top-K by strength)"
            )
        else:
            cluster_desc = "a strength-ranked sample (no embeddings available)"

        rep_lines = "\n".join(
            f"  - [{s.id}] strength={s.strength:.2f}: {s.content}"
            for s in samples
        )
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"{count_hint}"
            f"You see a consensus cluster forming in the shared signal "
            f"store. The cluster is {cluster_desc}, represented by "
            f"{len(samples)} signal(s):\n\n"
            f"{rep_lines}\n\n"
            f"Find a structural weakness that applies to the CLUSTER AS "
            f"A WHOLE — not to any individual signal. What shared "
            f"assumption do these signals make that might be wrong? "
            f"What kind of evidence would they all collectively miss? "
            f"One or two sentences.\n\n"
            f"{type_parent_instruction()}\n"
            f"OBJECTION:"
        )

    def parent_id_for_deposit(self, samples: list[Signal]):
        # link to the strongest representative as a structural anchor
        if not samples:
            return None
        return max(samples, key=lambda s: s.strength).id
