"""Stigmergic signal store - shared pheromone environment for swarm coordination."""

import time
import random
import asyncio
from threading import RLock as Lock  # RLock allows reentrant acquisition (deposit→get_ancestors)
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Signal:
    """A signal (pheromone) deposited by an agent."""
    id: str
    type: str  # OBSERVATION, INSIGHT, EVIDENCE, CRITIQUE, etc.
    content: str
    strength: float  # 0.0 to 1.0
    timestamp: float
    depositor: str
    parent: Optional[str] = None
    visits: int = 0  # Track sampling/usage (for exploration bonus)
    metadata: dict = None  # Additional metadata (source provenance, etc.)

    # Agent dialogue fields
    responses: List[str] = field(default_factory=list)  # IDs of direct responses
    is_response_to: Optional[str] = None  # ID of signal this responds to

    def __post_init__(self):
        """Ensure fields are initialized correctly."""
        self.strength = max(0.0, min(1.0, self.strength))
        if self.metadata is None:
            self.metadata = {}
        if self.responses is None:
            self.responses = []

    def get_parent_context(self) -> Optional[dict]:
        """Get cached parent context from metadata.

        Returns:
            Dict with parent_content, parent_type, parent_id or None
        """
        if self.metadata and "parent_content" in self.metadata:
            return {
                "content": self.metadata.get("parent_content"),
                "type": self.metadata.get("parent_type"),
                "id": self.parent
            }
        return None

    def get_provenance_chain(self) -> List[str]:
        """Get full ancestry chain of signal IDs.

        Returns:
            List of signal IDs from root to immediate parent
        """
        return self.metadata.get("provenance_chain", [self.parent] if self.parent else [])


class SignalStore:
    """Thread-safe pheromone environment for stigmergic coordination.

    Agents deposit signals with strength values. Strong signals attract
    more attention. Signals decay over time. Corroboration amplifies signals.
    """

    def __init__(self, decay_rate: float = 0.05, prune_threshold: float = 0.1,
                 diversity_threshold: float = 0.85, exploration_bonus: float = 0.3,
                 use_semantic_clustering: bool = True):
        """Initialize signal store.

        Args:
            decay_rate: Per-iteration strength reduction (0.0-1.0)
            prune_threshold: Remove signals below this strength
            diversity_threshold: Similarity threshold for duplicate detection (0.0-1.0)
            exploration_bonus: Bonus weight for under-visited signals (0.0-1.0)
            use_semantic_clustering: Use embeddings for semantic similarity (default: True)
        """
        self._lock = Lock()
        self.signals: Dict[str, Signal] = {}
        self.decay_rate = decay_rate
        self.prune_threshold = prune_threshold
        self.diversity_threshold = diversity_threshold
        self.exploration_bonus = exploration_bonus
        self._next_id = 0

        # Semantic clustering setup
        self.use_semantic_clustering = use_semantic_clustering
        self.embedding_model = None
        self.signal_embeddings: Dict[str, Any] = {}  # Signal ID -> embedding vector

        # FAISS index setup (for fast similarity search)
        self.use_faiss = use_semantic_clustering  # Enable FAISS with embeddings
        self.faiss_available = False
        self.faiss_indexes: Dict[str, Any] = {}  # signal_type -> FAISS index
        self.faiss_id_maps: Dict[str, List[str]] = {}  # signal_type -> [signal_ids]

        if use_semantic_clustering:
            self._initialize_embedding_model()
            self._initialize_faiss()

        # Graph traversal cache (performance optimization)
        self._ancestor_cache: Dict[tuple, List[Signal]] = {}
        self._descendant_cache: Dict[tuple, List[Signal]] = {}

        # PERFORMANCE: Signal indexes for O(1) lookup by type and parent
        # Replaces O(n) get_all_signals() + filter patterns
        self._signals_by_type: Dict[str, set] = {}  # signal_type -> {signal_ids}
        self._signals_by_parent: Dict[str, set] = {}  # parent_id -> {child_signal_ids}

        # Event-driven coordination: Agents wait for signals to be deposited
        self._signal_events: Dict[str, asyncio.Event] = {}  # signal_type -> Event
        self._new_signal_event = asyncio.Event()  # Triggered on ANY new signal

    def _initialize_embedding_model(self):
        """Initialize sentence transformer for semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading semantic embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Semantic clustering enabled (all-MiniLM-L6-v2)")
        except ImportError:
            logger.warning("sentence-transformers not available")
            logger.info("Install with: pip install sentence-transformers")
            logger.info("Falling back to string similarity")
            self.use_semantic_clustering = False
        except Exception as e:
            logger.warning("Could not load embedding model: %s", e)
            logger.info("Falling back to string similarity")
            self.use_semantic_clustering = False

    def _initialize_faiss(self):
        """Initialize FAISS for fast similarity search."""
        try:
            import faiss
            self.faiss_available = True
            logger.info("FAISS available for fast similarity search")
        except ImportError:
            self.faiss_available = False
            logger.warning("FAISS not available, using O(n) similarity search")
            logger.info("Install with: pip install faiss-cpu")
            self.use_faiss = False

    def _add_to_faiss_index(self, signal_id: str, signal_type: str, embedding: Any):
        """Add signal embedding to FAISS index for its type.

        Args:
            signal_id: Signal ID
            signal_type: Signal type (for separate indexes per type)
            embedding: Embedding vector (numpy array)
        """
        if not self.use_faiss or not self.faiss_available:
            return

        try:
            import faiss
            import numpy as np

            # Create index for this type if doesn't exist
            if signal_type not in self.faiss_indexes:
                embedding_dim = len(embedding)
                # Use IndexFlatIP for exact cosine similarity (after normalization)
                # This is O(n) but uses SIMD/optimized linear algebra (10-100x faster)
                index = faiss.IndexFlatIP(embedding_dim)
                self.faiss_indexes[signal_type] = index
                self.faiss_id_maps[signal_type] = []
                logger.debug("Created FAISS index for signal type: %s (dim=%d)",
                           signal_type, embedding_dim)

            # Convert to numpy array (embedding is already normalized at storage time)
            embedding = np.array(embedding, dtype=np.float32)

            # Add to index
            self.faiss_indexes[signal_type].add(embedding.reshape(1, -1))
            self.faiss_id_maps[signal_type].append(signal_id)

        except Exception as e:
            logger.warning("Could not add signal to FAISS index: %s", e)

    def _remove_from_faiss_index(self, signal_id: str, signal_type: str):
        """Remove signal from FAISS index (INTERNAL - assumes lock held).

        IMPORTANT: This method assumes self._lock is already held by the caller.
        It accesses and modifies self.faiss_id_maps which must be protected by the lock.

        FAISS doesn't support removal, so we mark as deleted in id_map.
        Rebuild index periodically when too many deleted.

        Args:
            signal_id: Signal ID to remove
            signal_type: Signal type
        """
        if not self.use_faiss or signal_type not in self.faiss_id_maps:
            return

        try:
            # Find signal in id map and mark as deleted
            id_map = self.faiss_id_maps[signal_type]
            if signal_id in id_map:
                idx = id_map.index(signal_id)
                id_map[idx] = None  # Mark as deleted

                # Rebuild if >30% deleted (amortized O(1))
                deleted_count = id_map.count(None)
                if len(id_map) > 0 and deleted_count / len(id_map) > 0.3:
                    self._rebuild_faiss_index(signal_type)

        except Exception as e:
            logger.warning("Could not remove signal from FAISS index: %s", e)

    def _rebuild_faiss_index(self, signal_type: str):
        """Rebuild FAISS index for a signal type (remove deleted entries).

        IMPORTANT: This method assumes self._lock is already held by the caller.
        It accesses self.signals, self.signal_embeddings, self.faiss_indexes,
        and self.faiss_id_maps which must all be protected by the lock.

        Args:
            signal_type: Signal type to rebuild
        """
        if signal_type not in self.faiss_id_maps:
            return

        try:
            import faiss
            import numpy as np

            # Get non-deleted signals (defensive copy to avoid iteration issues)
            id_map = self.faiss_id_maps[signal_type]
            # Make a copy of the list to avoid issues if dict changes during iteration
            id_map_snapshot = list(id_map)
            valid_ids = [sid for sid in id_map_snapshot
                        if sid is not None and sid in self.signals]

            if not valid_ids:
                # No valid signals, clear index
                del self.faiss_indexes[signal_type]
                del self.faiss_id_maps[signal_type]
                logger.debug("Cleared empty FAISS index for type: %s", signal_type)
                return

            # Get embeddings for valid signals
            embeddings = []
            new_id_map = []
            for sid in valid_ids:
                if sid in self.signal_embeddings:
                    emb = np.array(self.signal_embeddings[sid], dtype=np.float32)
                    # NORMALIZATION: Handle both legacy (unnormalized) and new (normalized) embeddings
                    # New embeddings are already normalized, but we normalize here for safety
                    # to handle any legacy embeddings from before the normalization fix
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm  # Normalize (idempotent for already-normalized embeddings)
                        embeddings.append(emb)
                        new_id_map.append(sid)

            if not embeddings:
                del self.faiss_indexes[signal_type]
                del self.faiss_id_maps[signal_type]
                return

            # Create new index
            embedding_dim = len(embeddings[0])
            index = faiss.IndexFlatIP(embedding_dim)
            index.add(np.array(embeddings, dtype=np.float32))

            # Update
            self.faiss_indexes[signal_type] = index
            self.faiss_id_maps[signal_type] = new_id_map

            logger.debug("Rebuilt FAISS index for %s: %d signals", signal_type, len(new_id_map))

        except Exception as e:
            logger.warning("Could not rebuild FAISS index: %s", e)

    def _find_similar_faiss(self, signal_type: str, embedding: Any,
                            similarity_threshold: float, max_results: int = 100) -> List[str]:
        """Find similar signals using FAISS index.

        Args:
            signal_type: Signal type to search
            embedding: Query embedding (will be normalized)
            similarity_threshold: Minimum cosine similarity
            max_results: Maximum results to return

        Returns:
            List of signal IDs above similarity threshold
        """
        if (not self.use_faiss or
            signal_type not in self.faiss_indexes or
            not self.faiss_id_maps[signal_type]):
            return []

        try:
            import numpy as np

            # Normalize query embedding
            query = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

            # Search FAISS index
            # Since we use IndexFlatIP with normalized vectors, scores are cosine similarities
            k = min(max_results, len(self.faiss_id_maps[signal_type]))
            if k == 0:
                return []

            scores, indices = self.faiss_indexes[signal_type].search(
                query.reshape(1, -1), k
            )

            # Filter by threshold and map back to signal IDs
            similar_ids = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= similarity_threshold:
                    signal_id = self.faiss_id_maps[signal_type][idx]
                    if signal_id is not None and signal_id in self.signals:
                        similar_ids.append(signal_id)

            return similar_ids

        except Exception as e:
            logger.warning("FAISS search failed: %s", e)
            return []

    def _check_similarity(self, content1: str, content2: str,
                         embedding1: Any = None, embedding2: Any = None) -> float:
        """Check similarity between two signal contents.

        Uses semantic embeddings if available, otherwise falls back to string matching.

        Args:
            content1: First signal content
            content2: Second signal content
            embedding1: Optional pre-computed embedding for content1
            embedding2: Optional pre-computed embedding for content2

        Returns:
            Similarity ratio (0.0-1.0)
        """
        # Use semantic similarity if embeddings available
        if self.use_semantic_clustering and self.embedding_model is not None:
            import numpy as np

            if embedding1 is None:
                embedding1 = self.embedding_model.encode(content1)
                # Normalize on-the-fly computed embeddings
                emb1_array = np.array(embedding1, dtype=np.float32)
                norm1 = np.linalg.norm(emb1_array)
                if norm1 > 0:
                    embedding1 = emb1_array / norm1

            if embedding2 is None:
                embedding2 = self.embedding_model.encode(content2)
                # Normalize on-the-fly computed embeddings
                emb2_array = np.array(embedding2, dtype=np.float32)
                norm2 = np.linalg.norm(emb2_array)
                if norm2 > 0:
                    embedding2 = emb2_array / norm2

            # Cosine similarity (embeddings are normalized, so just dot product)
            # But we still normalize to handle legacy embeddings
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)

        # Fallback to string similarity
        return SequenceMatcher(None, content1.lower(), content2.lower()).ratio()

    def deposit(self, signal_type: str, content: str, strength: float,
                depositor: str, parent: Optional[str] = None, metadata: Optional[dict] = None) -> Optional[str]:
        """Deposit a new signal in the environment with diversity checking.

        Args:
            signal_type: Type of signal (OBSERVATION, INSIGHT, EVIDENCE, CRITIQUE, etc.)
            content: Signal content
            strength: Initial strength (0.0-1.0)
            depositor: Agent ID that deposited
            parent: Optional parent signal ID
            metadata: Optional metadata dictionary (provenance, etc.)

        Returns:
            Signal ID if deposited, None if rejected as duplicate
        """
        with self._lock:
            # Check for near-duplicate signals of same type
            # OPTIMIZATION: Only check recent signals (last 5 minutes)
            # Duplicates are typically deposited close in time, old signals likely decayed
            recent_cutoff = time.time() - 300  # 5 minutes
            same_type = [s for s in self.signals.values()
                        if s.type == signal_type and s.timestamp > recent_cutoff]

            # OPTIMIZATION: Only compute embedding if there are signals to compare against
            new_embedding = None
            if same_type and self.use_semantic_clustering and self.embedding_model is not None:
                import numpy as np
                new_embedding = self.embedding_model.encode(content)
                # NORMALIZATION: Store normalized embeddings for consistency with FAISS
                # This avoids repeated normalization during similarity checks
                embedding_array = np.array(new_embedding, dtype=np.float32)
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    new_embedding = (embedding_array / norm).tolist()
                else:
                    new_embedding = embedding_array.tolist()

            # Use FAISS for fast similarity search (10-100x faster than O(n) loop)
            if self.use_faiss and self.faiss_available and new_embedding is not None:
                similar_ids = self._find_similar_faiss(
                    signal_type, new_embedding, self.diversity_threshold
                )
                # Check each similar signal (filtered by FAISS)
                for similar_id in similar_ids:
                    existing = self.signals.get(similar_id)
                    if existing and existing.timestamp > recent_cutoff:
                        # FAISS already confirmed similarity >= threshold
                        existing.strength = min(1.0, existing.strength * 1.1)
                        existing.visits += 1
                        logger.info("Rejected duplicate %s via FAISS, amplified %s",
                                   signal_type, existing.id)
                        return None
            else:
                # Fallback: O(n) search (original implementation)
                for existing in same_type:
                    existing_embedding = self.signal_embeddings.get(existing.id)
                    similarity = self._check_similarity(
                        content, existing.content,
                        embedding1=new_embedding,
                        embedding2=existing_embedding
                    )
                    if similarity >= self.diversity_threshold:
                        # Too similar - amplify existing instead of creating duplicate
                        existing.strength = min(1.0, existing.strength * 1.1)
                        existing.visits += 1
                        logger.info("Rejected duplicate %s (similarity: %.2f), amplified %s",
                                   signal_type, similarity, existing.id)
                        return None

            signal_id = f"{signal_type}_{self._next_id:04d}"
            self._next_id += 1

            # PROVENANCE BOOST: Check if parent has verified ancestry
            boost_applied = False
            if parent and parent in self.signals:
                verifications = self.get_ancestors(parent, target_type="VERIFICATION")
                if verifications:
                    # Calculate average verification confidence
                    avg_conf = sum(v.strength for v in verifications) / len(verifications)

                    # Boost signal if ancestry is highly verified (>0.7 confidence)
                    if avg_conf >= 0.7:
                        boost_factor = 1 + (0.2 * avg_conf)  # Up to 20% boost
                        original_strength = strength
                        strength = min(1.0, strength * boost_factor)

                        # Track in metadata
                        if metadata is None:
                            metadata = {}
                        metadata['provenance_boost'] = boost_factor
                        metadata['verified_ancestors'] = len(verifications)
                        metadata['original_strength'] = original_strength
                        boost_applied = True

            signal = Signal(
                id=signal_id,
                type=signal_type,
                content=content,
                strength=strength,
                timestamp=time.time(),
                depositor=depositor,
                parent=parent,
                metadata=metadata or {}
            )

            self.signals[signal_id] = signal

            # PERFORMANCE: Update indexes for fast lookup
            if signal_type not in self._signals_by_type:
                self._signals_by_type[signal_type] = set()
            self._signals_by_type[signal_type].add(signal_id)

            if parent:
                if parent not in self._signals_by_parent:
                    self._signals_by_parent[parent] = set()
                self._signals_by_parent[parent].add(signal_id)

            # Log provenance boost (after signal creation)
            if boost_applied:
                boost_factor = metadata['provenance_boost']
                verified_count = metadata['verified_ancestors']
                logger.info("Signal %s boosted by %.2fx (verified ancestors: %d, avg_conf: %.2f)",
                          signal_id, boost_factor, verified_count, avg_conf)

            # Store embedding and add to FAISS index
            if new_embedding is not None:
                self.signal_embeddings[signal_id] = new_embedding
                # Add to FAISS index for fast similarity search
                self._add_to_faiss_index(signal_id, signal_type, new_embedding)

            # OPTIMIZATION: Selective cache invalidation instead of full clear
            # Only invalidate cache entries involving this signal or its ancestors
            self._invalidate_cache_for_signal(signal_id, parent)

            # EVENT-DRIVEN: Notify waiting agents that a signal was deposited
            # Create event for this signal type if it doesn't exist
            if signal_type not in self._signal_events:
                self._signal_events[signal_type] = asyncio.Event()

            # Set events to wake up waiting agents
            self._signal_events[signal_type].set()
            self._new_signal_event.set()

            return signal_id

    def deposit_response(self, signal_type: str, content: str, strength: float,
                        depositor: str, responding_to: str,
                        metadata: Optional[dict] = None) -> Optional[str]:
        """Deposit a signal that directly responds to another signal.

        Args:
            signal_type: Type of signal (INSIGHT, OBJECTION, etc.)
            content: Signal content
            strength: Initial strength (0.0-1.0)
            depositor: Agent ID that deposited
            responding_to: Signal ID this responds to
            metadata: Optional metadata dictionary

        Returns:
            Signal ID if deposited, None if rejected or target doesn't exist
        """
        # Verify responding_to signal exists (before deposit to avoid deadlock)
        target = self.get_signal(responding_to)
        if not target:
            logger.debug("Cannot respond to non-existent signal: %s", responding_to)
            return None

        # Deposit signal with target as parent (deposit has its own lock)
        signal_id = self.deposit(
            signal_type=signal_type,
            content=content,
            strength=strength,
            depositor=depositor,
            parent=responding_to,  # Parent link for provenance
            metadata=metadata
        )

        if signal_id:
            # Update response links (need lock here)
            with self._lock:
                # Update target's response list
                target.responses.append(signal_id)

                # Mark this signal as a response
                response_signal = self.signals[signal_id]
                response_signal.is_response_to = responding_to

            logger.debug("%s responded to %s with %s", depositor, responding_to, signal_id)

        return signal_id

    def get_responses(self, signal_id: str) -> List[Signal]:
        """Get all direct responses to a signal.

        Args:
            signal_id: Signal ID to get responses for

        Returns:
            List of direct response signals (empty if none)
        """
        with self._lock:
            signal = self.signals.get(signal_id)
            if not signal:
                return []

            return [self.signals[resp_id] for resp_id in signal.responses
                    if resp_id in self.signals]

    def get_dialogue_thread(self, signal_id: str, max_depth: int = 10) -> List[Signal]:
        """Get full dialogue thread (signal + all responses recursively).

        CYCLE DETECTION: Prevents infinite loops if circular response references exist.
        Uses visited set to track already-processed signals.

        Args:
            signal_id: Starting signal ID
            max_depth: Maximum recursion depth

        Returns:
            List of signals in dialogue thread (ordered by traversal)
        """
        with self._lock:
            thread = []
            visited = set()  # Track visited signals to prevent cycles

            def collect_thread(sig_id: str, depth: int):
                """Recursively collect dialogue thread with cycle detection."""
                # Stop conditions: max depth, already visited (cycle), or invalid signal
                if depth >= max_depth or sig_id in visited:
                    return

                sig = self.signals.get(sig_id)
                if not sig:
                    return

                visited.add(sig_id)  # Mark as visited before recursing
                thread.append(sig)

                # Recursively get responses
                for resp_id in sig.responses:
                    collect_thread(resp_id, depth + 1)

            collect_thread(signal_id, 0)
            return thread

    def get_signal(self, signal_id: str) -> Optional[Signal]:
        """Get a signal by ID.

        Args:
            signal_id: Signal ID to retrieve

        Returns:
            Signal object or None if not found
        """
        with self._lock:
            return self.signals.get(signal_id)

    async def wait_for_signal(self, signal_type: str, timeout: float = None) -> bool:
        """Wait for a signal of specified type to be deposited.

        Args:
            signal_type: Signal type to wait for
            timeout: Optional timeout in seconds (None = wait forever)

        Returns:
            True if signal was deposited, False if timeout
        """
        # Create event for this type if it doesn't exist
        if signal_type not in self._signal_events:
            self._signal_events[signal_type] = asyncio.Event()

        event = self._signal_events[signal_type]

        try:
            if timeout:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            else:
                await event.wait()
            return True
        except asyncio.TimeoutError:
            return False

    def clear_signal_event(self, signal_type: str):
        """Clear event after agents have been notified.

        Args:
            signal_type: Signal type to clear event for
        """
        if signal_type in self._signal_events:
            self._signal_events[signal_type].clear()

    def has_signals(self, signal_type: str) -> bool:
        """Check if any signals of type exist.

        Args:
            signal_type: Signal type to check

        Returns:
            True if signals exist
        """
        with self._lock:
            return any(s.type == signal_type for s in self.signals.values())

    def sample_weighted(self, signal_type: str, n: int = 1) -> List[Signal]:
        """Sample n signals probabilistically weighted by strength with exploration bonus.

        Signals with fewer visits get bonus weight to encourage exploration of
        dissenting/under-explored signals.

        Args:
            signal_type: Type of signal to sample
            n: Number of signals to sample

        Returns:
            List of sampled signals (may be fewer than n if not enough exist)
        """
        with self._lock:
            # Filter by type
            candidates = [s for s in self.signals.values() if s.type == signal_type]

            if not candidates:
                return []

            # Calculate max visits for normalization
            max_visits = max(s.visits for s in candidates) if candidates else 1
            max_visits = max(max_visits, 1)  # Avoid division by zero

            # Weight by strength + exploration bonus for under-visited signals
            weights = []
            for s in candidates:
                # Base weight from strength
                base_weight = s.strength + 0.1

                # Exploration bonus: higher bonus for fewer visits
                visit_ratio = s.visits / max_visits
                exploration_weight = self.exploration_bonus * (1.0 - visit_ratio)

                total_weight = base_weight + exploration_weight
                weights.append(total_weight)

            total_weight = sum(weights)

            # Normalize
            probabilities = [w / total_weight for w in weights]

            # Sample without replacement up to n
            k = min(n, len(candidates))
            sampled = random.choices(candidates, weights=probabilities, k=k)

            # INCREMENT VISITS: Track that these signals were sampled (for exploration bonus)
            for signal in sampled:
                signal.visits += 1

            return sampled

    def sample_stratified(
        self,
        signal_type: str,
        weak: int = 0,
        medium: int = 0,
        strong: int = 0,
        weak_threshold: float = 0.4,
        strong_threshold: float = 0.7
    ) -> List[Signal]:
        """Sample signals stratified by strength (weak/medium/strong).

        This enables balanced sampling across quality levels instead of bias toward
        strong signals. Useful for critics who need to evaluate signals at all quality
        levels, not just the top performers.

        Args:
            signal_type: Type of signals to sample
            weak: Number of weak signals to sample (strength < weak_threshold)
            medium: Number of medium signals to sample (weak_threshold <= strength < strong_threshold)
            strong: Number of strong signals to sample (strength >= strong_threshold)
            weak_threshold: Strength threshold for weak signals (default: 0.4)
            strong_threshold: Strength threshold for strong signals (default: 0.7)

        Returns:
            List of sampled signals from each stratum (may be fewer if not enough exist)

        Example:
            # Sample 2 weak, 2 medium, 2 strong signals for balanced evaluation
            signals = store.sample_stratified("INITIAL", weak=2, medium=2, strong=2)
        """
        with self._lock:
            # Filter by type
            candidates = [s for s in self.signals.values() if s.type == signal_type]

            if not candidates:
                return []

            # Stratify into weak/medium/strong
            weak_signals = [s for s in candidates if s.strength < weak_threshold]
            medium_signals = [s for s in candidates
                            if weak_threshold <= s.strength < strong_threshold]
            strong_signals = [s for s in candidates if s.strength >= strong_threshold]

            result = []

            # Sample from weak stratum
            if weak > 0 and weak_signals:
                k = min(weak, len(weak_signals))
                result.extend(random.sample(weak_signals, k))

            # Sample from medium stratum
            if medium > 0 and medium_signals:
                k = min(medium, len(medium_signals))
                result.extend(random.sample(medium_signals, k))

            # Sample from strong stratum
            if strong > 0 and strong_signals:
                k = min(strong, len(strong_signals))
                result.extend(random.sample(strong_signals, k))

            return result

    def amplify(self, signal_id: str, factor: float = 1.2) -> bool:
        """Amplify signal strength (corroboration).

        Args:
            signal_id: ID of signal to amplify
            factor: Multiplicative factor (e.g., 1.2 = 20% increase)

        Returns:
            True if signal found and amplified
        """
        with self._lock:
            if signal_id in self.signals:
                signal = self.signals[signal_id]
                signal.strength = min(1.0, signal.strength * factor)
                signal.visits += 1
                return True
            return False

    def boost_contrarian_signals(self, contrarian_types: List[str], boost_factor: float = 1.15):
        """Apply boost to contrarian signal types to offset echo chamber effects.

        Args:
            contrarian_types: List of signal types to boost (e.g., ["CRITIQUE", "COUNTER_EVIDENCE"])
            boost_factor: Multiplicative boost factor (e.g., 1.15 = 15% increase)

        Returns:
            Number of signals boosted
        """
        with self._lock:
            boosted = 0
            for signal in self.signals.values():
                if signal.type in contrarian_types:
                    signal.strength = min(1.0, signal.strength * boost_factor)
                    boosted += 1
            return boosted

    def decay_all(self, contrarian_types: Optional[List[str]] = None, contrarian_boost: float = 1.10) -> int:
        """Apply decay to all signals (evaporation) with optional contrarian boost.

        Args:
            contrarian_types: Optional list of signal types to receive anti-decay boost
            contrarian_boost: Multiplier for contrarian signals (default: 1.10 = 10% boost)

        Returns:
            Number of signals remaining after decay
        """
        with self._lock:
            for signal in self.signals.values():
                # Apply decay
                signal.strength *= (1.0 - self.decay_rate)

                # Apply contrarian boost to offset echo effects
                if contrarian_types and signal.type in contrarian_types:
                    signal.strength = min(1.0, signal.strength * contrarian_boost)

            return len(self.signals)

    def _delete_signal_locked(self, signal_id: str) -> bool:
        """Delete a signal and all associated data (INTERNAL - assumes lock held).

        IMPORTANT: This method assumes self._lock is already held by the caller.
        For external use, call delete_signal() instead.

        Ensures cleanup of:
        - Signal from signals dict
        - Embedding from signal_embeddings dict
        - Cache entries involving this signal

        Args:
            signal_id: Signal ID to delete

        Returns:
            True if signal was deleted, False if not found
        """
        if signal_id not in self.signals:
            return False

        # Get signal info before deletion (for cache cleanup and FAISS removal)
        signal = self.signals[signal_id]
        signal_type = signal.type
        parent = signal.parent

        # Delete signal
        del self.signals[signal_id]

        # PERFORMANCE: Remove from indexes
        if signal_type in self._signals_by_type:
            self._signals_by_type[signal_type].discard(signal_id)
            # Clean up empty sets
            if not self._signals_by_type[signal_type]:
                del self._signals_by_type[signal_type]

        if parent and parent in self._signals_by_parent:
            self._signals_by_parent[parent].discard(signal_id)
            # Clean up empty sets
            if not self._signals_by_parent[parent]:
                del self._signals_by_parent[parent]

        # Delete embedding (prevent memory leak)
        if signal_id in self.signal_embeddings:
            del self.signal_embeddings[signal_id]

        # Remove from FAISS index (prevent stale index)
        self._remove_from_faiss_index(signal_id, signal_type)

        # Clean up cache entries (prevent stale cache)
        # Remove ancestor cache entries
        keys_to_remove = [k for k in self._ancestor_cache.keys()
                         if k[0] == signal_id]
        for key in keys_to_remove:
            del self._ancestor_cache[key]

        # Remove descendant cache entries
        keys_to_remove = [k for k in self._descendant_cache.keys()
                         if k[0] == signal_id]
        for key in keys_to_remove:
            del self._descendant_cache[key]

        return True

    def delete_signal(self, signal_id: str) -> bool:
        """Delete a signal and all associated data (centralized deletion method).

        Ensures cleanup of:
        - Signal from signals dict
        - Embedding from signal_embeddings dict
        - Cache entries involving this signal

        This is the ONLY safe way to delete signals - prevents memory leaks.

        Args:
            signal_id: Signal ID to delete

        Returns:
            True if signal was deleted, False if not found
        """
        with self._lock:
            return self._delete_signal_locked(signal_id)

    def prune_weak(self) -> int:
        """Remove signals below pruning threshold.

        Also cleans up associated embeddings to prevent memory leaks.

        Returns:
            Number of signals pruned
        """
        with self._lock:
            # RACE CONDITION FIX: Hold lock during entire operation
            # Identify signals to remove
            to_remove = [
                sid for sid, signal in self.signals.items()
                if signal.strength < self.prune_threshold
            ]

            # Delete signals while holding lock (using internal locked method)
            pruned_count = 0
            for sid in to_remove:
                if self._delete_signal_locked(sid):
                    pruned_count += 1

        return pruned_count

    def get_top_signals(self, signal_type: str, n: int = 10) -> List[Signal]:
        """Get top n strongest signals of a type.

        Args:
            signal_type: Type of signal
            n: Number to return

        Returns:
            List of signals sorted by strength (descending)
        """
        with self._lock:
            candidates = [s for s in self.signals.values() if s.type == signal_type]
            candidates.sort(key=lambda s: s.strength, reverse=True)
            return candidates[:n]

    def get_all_signals(self) -> List[Signal]:
        """Get all signals in the store.

        PERFORMANCE NOTE: Prefer get_signals_by_type() or get_signals_by_parent()
        for filtered queries - they use indexes for O(k) vs O(n) performance.

        Returns:
            List of all signals
        """
        with self._lock:
            return list(self.signals.values())

    def get_signals_by_type(self, signal_type: str) -> List[Signal]:
        """Get all signals of a specific type using indexed lookup.

        PERFORMANCE: O(k) where k = number of matching signals (vs O(n) for get_all_signals + filter)

        Args:
            signal_type: Signal type to retrieve (INSIGHT, CRITIQUE, VERIFICATION, etc.)

        Returns:
            List of signals of the specified type (empty list if none)
        """
        with self._lock:
            signal_ids = self._signals_by_type.get(signal_type, set())
            return [self.signals[sid] for sid in signal_ids if sid in self.signals]

    def get_signals_by_parent(self, parent_id: str) -> List[Signal]:
        """Get all signals that are children of a specific parent using indexed lookup.

        PERFORMANCE: O(k) where k = number of children (vs O(n) for get_all_signals + filter)

        Args:
            parent_id: Parent signal ID

        Returns:
            List of child signals (empty list if none)
        """
        with self._lock:
            child_ids = self._signals_by_parent.get(parent_id, set())
            return [self.signals[sid] for sid in child_ids if sid in self.signals]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about signal store.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            by_type = {}
            for signal in self.signals.values():
                by_type[signal.type] = by_type.get(signal.type, 0) + 1

            strengths = [s.strength for s in self.signals.values()]
            avg_strength = sum(strengths) / len(strengths) if strengths else 0.0

            return {
                "total_signals": len(self.signals),
                "by_type": by_type,
                "avg_strength": avg_strength,
                "strongest": max(strengths) if strengths else 0.0
            }

    def clear(self) -> None:
        """Clear all signals and associated embeddings."""
        with self._lock:
            self.signals.clear()
            self.signal_embeddings.clear()  # BUGFIX: Clear embeddings too
            self._next_id = 0

    # ========================================================================
    # GRAPH OPERATIONS FOR DOCUMENT PROCESSING
    # ========================================================================

    def find_related_signals(self, signal: Signal, type: str,
                           similarity_threshold: float = 0.5,
                           n: int = 5) -> List[Signal]:
        """Find signals similar to a given signal.

        Used by critics, monitors, and haters to find related signals for
        consistency checking, consensus detection, and cluster analysis.

        Args:
            signal: The signal to find related signals for
            type: Type of signals to search (e.g., "INSIGHT")
            similarity_threshold: Minimum similarity (0.0-1.0)
            n: Maximum number of related signals to return

        Returns:
            List of related signals (excluding the input signal itself)
        """
        with self._lock:
            # Get candidates of the specified type (excluding the input signal)
            candidates = [s for s in self.signals.values()
                         if s.type == type and s.id != signal.id]

            if not candidates:
                return []

            # Get embedding for the input signal
            signal_embedding = self.signal_embeddings.get(signal.id)

            # Calculate similarities
            similarities = []
            for candidate in candidates:
                candidate_embedding = self.signal_embeddings.get(candidate.id)
                sim = self._check_similarity(
                    signal.content, candidate.content,
                    embedding1=signal_embedding,
                    embedding2=candidate_embedding
                )
                if sim >= similarity_threshold:
                    similarities.append((candidate, sim))

            # Sort by similarity (highest first) and return top n
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [candidate for candidate, sim in similarities[:n]]

    def sample_cluster(self, signal_type: str, size: int = 3,
                      similarity_threshold: float = 0.4) -> List[Signal]:
        """Sample a cluster of related signals for pattern finding.

        Uses semantic embeddings when available for better cross-domain pattern discovery.
        Falls back to string similarity if embeddings unavailable.

        This enables foragers to explore combinations of observations from
        different parts of the corpus to discover patterns.

        Args:
            signal_type: Type of signals to cluster (e.g., "OBSERVATION")
            size: Number of signals in cluster (3-5 recommended)
            similarity_threshold: Minimum similarity between signals (0.0-1.0)
                                 For semantic: 0.6-0.7 recommended
                                 For string: 0.3-0.4 recommended

        Returns:
            List of related signals (may be fewer than size if not enough exist)
        """
        with self._lock:
            candidates = [s for s in self.signals.values() if s.type == signal_type]

            if len(candidates) < 2:
                return candidates

            # Pick random seed signal (or could pick highest strength for quality)
            seed = random.choice(candidates)
            cluster = [seed]
            seed_embedding = self.signal_embeddings.get(seed.id)

            # Find similar signals using semantic or string similarity
            similarities = []
            for candidate in candidates:
                if candidate.id != seed.id:
                    candidate_embedding = self.signal_embeddings.get(candidate.id)
                    sim = self._check_similarity(
                        seed.content, candidate.content,
                        embedding1=seed_embedding,
                        embedding2=candidate_embedding
                    )
                    if sim >= similarity_threshold:
                        similarities.append((candidate, sim))

            # Sort by similarity and take top N-1
            similarities.sort(key=lambda x: x[1], reverse=True)
            for candidate, sim in similarities[:size-1]:
                cluster.append(candidate)

            return cluster

    def get_validation_status(self, signal_id: str) -> dict:
        """Get comprehensive validation metrics for a signal.

        Used by critics to evaluate how well an insight is supported.

        Args:
            signal_id: Signal ID to evaluate

        Returns:
            Dictionary with validation metrics:
            - evidence_count: Number of EVIDENCE children
            - observation_count: Number of OBSERVATION ancestors
            - critique_count: Number of CRITIQUE children
            - validation_score: Overall score 0.0-1.0
            - has_contradictions: Boolean if challenged
        """
        with self._lock:
            if signal_id not in self.signals:
                return {
                    'evidence_count': 0,
                    'observation_count': 0,
                    'critique_count': 0,
                    'validation_score': 0.0,
                    'has_contradictions': False
                }

            signal = self.signals[signal_id]

            # Count evidence children (supporting signals)
            evidence_count = sum(
                1 for s in self.signals.values()
                if s.parent == signal_id and s.type == "EVIDENCE"
            )

            # Count observation ancestors (how many sources)
            observation_ancestors = self.get_ancestors(signal_id, target_type="OBSERVATION")
            observation_count = len(observation_ancestors)

            # CRITICAL FIX: Foragers put additional observations in metadata
            # Check metadata['observation_ids'] as fallback to catch all sources
            if 'observation_ids' in signal.metadata:
                metadata_obs_ids = signal.metadata['observation_ids']
                # Use the larger count (metadata is more complete for multi-source insights)
                metadata_count = len(metadata_obs_ids)
                if metadata_count > observation_count:
                    observation_count = metadata_count

            # Count critique children (challenges)
            critique_count = sum(
                1 for s in self.signals.values()
                if s.parent == signal_id and s.type == "CRITIQUE"
            )

            # Check for contradictions
            has_contradictions = any(
                'contradict' in s.content.lower() or 'disagree' in s.content.lower()
                for s in self.signals.values()
                if s.parent == signal_id and s.type in ["CRITIQUE", "OBJECTION"]
            )

            # Calculate validation score
            # Well-validated: 2+ evidence, 3+ observations, no contradictions
            score = 0.0
            if evidence_count >= 2:
                score += 0.4
            elif evidence_count >= 1:
                score += 0.2

            if observation_count >= 3:
                score += 0.3
            elif observation_count >= 2:
                score += 0.2
            elif observation_count >= 1:
                score += 0.1

            if critique_count > 0 and not has_contradictions:
                # Survived critique = stronger
                score += 0.2
            elif has_contradictions:
                # Contradicted = weaker
                score -= 0.3

            validation_score = max(0.0, min(1.0, score))

            return {
                'evidence_count': evidence_count,
                'observation_count': observation_count,
                'critique_count': critique_count,
                'validation_score': validation_score,
                'has_contradictions': has_contradictions
            }

    def get_unvalidated_signals(self, signal_type: str,
                               min_evidence: int = 2,
                               prioritize: bool = True) -> List[Signal]:
        """Get signals lacking sufficient validation, sorted by priority.

        Used by gatherers to find insights needing external validation.
        Prioritizes high-value insights (multi-source, high-strength) for
        efficient validation resource allocation.

        Args:
            signal_type: Type of signals to check (e.g., "INSIGHT")
            min_evidence: Minimum evidence count for validation
            prioritize: If True, sort by priority score (high-value first)

        Returns:
            List of signals needing validation (prioritized if requested)
        """
        with self._lock:
            unvalidated = []

            for signal in self.signals.values():
                if signal.type != signal_type:
                    continue

                # Count evidence children
                evidence_count = sum(
                    1 for s in self.signals.values()
                    if s.parent == signal.id and s.type == "EVIDENCE"
                )

                if evidence_count < min_evidence:
                    # Calculate priority score for this insight
                    obs_count = len(signal.metadata.get('observation_ids', []))
                    source_count = len(signal.metadata.get('source_documents', []))

                    # Priority factors:
                    # - Current strength (40%)
                    # - Observation count (30%) - multi-source insights are high value
                    # - Document diversity (20%) - cross-document patterns are high value
                    # - Novelty (10%) - under-visited signals get a boost
                    priority = (
                        signal.strength * 0.4 +
                        min(obs_count / 10.0, 1.0) * 0.3 +
                        min(source_count / 5.0, 1.0) * 0.2 +
                        (1.0 / (signal.visits + 1)) * 0.1
                    )

                    unvalidated.append((priority, signal))

            # Sort by priority if requested
            if prioritize:
                unvalidated.sort(key=lambda x: x[0], reverse=True)

            # Return just the signals (strip priority scores)
            return [sig for _, sig in unvalidated]

    def _invalidate_cache_for_signal(self, signal_id: str, parent_id: Optional[str]):
        """Selectively invalidate cache entries affected by a new signal.

        When a new signal is added, we only need to invalidate:
        1. Descendant caches for the parent and all its ancestors
        2. Ancestor caches that might query this new signal

        This is much more efficient than clearing the entire cache.

        Args:
            signal_id: The newly added signal ID
            parent_id: The parent signal ID (if any)
        """
        if not parent_id:
            # No parent means no ancestor chain to invalidate
            return

        # Invalidate descendant caches for parent and all ancestors
        # (they now have a new descendant)
        current = parent_id
        while current and current in self.signals:
            # Remove all cache entries for this ancestor (all target_type variants)
            keys_to_remove = [k for k in self._descendant_cache.keys() if k[0] == current]
            for key in keys_to_remove:
                del self._descendant_cache[key]

            # Move up the chain
            current = self.signals[current].parent

        # Note: We don't need to invalidate ancestor cache because:
        # - The new signal has no children yet (so no one will query its ancestors)
        # - Existing signals' ancestor chains haven't changed

    def get_ancestors(self, signal_id: str, target_type: Optional[str] = None) -> List[Signal]:
        """Get all ancestor signals by traversing parent links.

        Uses caching for performance - cache is invalidated on deposit.

        Args:
            signal_id: Starting signal ID
            target_type: If specified, only return ancestors of this type

        Returns:
            List of ancestor signals
        """
        # Check cache
        cache_key = (signal_id, target_type)
        if cache_key in self._ancestor_cache:
            return self._ancestor_cache[cache_key]

        with self._lock:
            ancestors = []
            visited = set()
            queue = [signal_id]

            while queue:
                current_id = queue.pop(0)

                if current_id in visited or current_id not in self.signals:
                    continue

                visited.add(current_id)
                signal = self.signals[current_id]

                # Add to ancestors if matches target type (or no filter)
                if current_id != signal_id:  # Don't include self
                    if target_type is None or signal.type == target_type:
                        ancestors.append(signal)

                # Traverse to parent
                if signal.parent:
                    queue.append(signal.parent)

            # Cache result
            self._ancestor_cache[cache_key] = ancestors
            return ancestors

    def get_descendants(self, signal_id: str, target_type: Optional[str] = None) -> List[Signal]:
        """Get all descendant signals by traversing child links.

        Uses caching for performance - cache is invalidated on deposit.

        Args:
            signal_id: Starting signal ID
            target_type: If specified, only return descendants of this type

        Returns:
            List of descendant signals
        """
        # Check cache
        cache_key = (signal_id, target_type)
        if cache_key in self._descendant_cache:
            return self._descendant_cache[cache_key]

        with self._lock:
            descendants = []
            visited = set()
            queue = [signal_id]

            while queue:
                current_id = queue.pop(0)

                if current_id in visited:
                    continue

                visited.add(current_id)

                # Find children
                children = [
                    s for s in self.signals.values()
                    if s.parent == current_id
                ]

                for child in children:
                    if target_type is None or child.type == target_type:
                        descendants.append(child)
                    queue.append(child.id)

            # Cache result
            self._descendant_cache[cache_key] = descendants
            return descendants

    def get_children(self, signal_id: str, child_type: Optional[str] = None) -> List[Signal]:
        """Get immediate child signals (direct descendants only).

        Args:
            signal_id: Parent signal ID
            child_type: If specified, only return children of this type

        Returns:
            List of child signals sorted by strength (descending)
        """
        with self._lock:
            children = [
                s for s in self.signals.values()
                if s.parent == signal_id
            ]

            # Filter by type if specified
            if child_type:
                children = [c for c in children if c.type == child_type]

            # Sort by strength (strongest first)
            children.sort(key=lambda s: s.strength, reverse=True)

            return children


# ============================================================================
# ENHANCED EVIDENCE CHAINING - Helper Functions
# ============================================================================

def deposit_with_context(signal_store: SignalStore, signal_type: str, content: str,
                        strength: float, depositor: str,
                        parent_signal: Optional[Signal] = None) -> Optional[str]:
    """Deposit signal with enhanced parent context for better evidence chaining.

    This helper enriches the signal metadata with parent context, provenance chains,
    and validation. It ensures proper evidence chaining across generations.

    Args:
        signal_store: SignalStore to deposit into
        signal_type: Type of signal (INITIAL, SUPPORT, CRITIQUE, OBJECTION)
        content: Signal content
        strength: Initial strength (0.0-1.0)
        depositor: Agent ID that deposited
        parent_signal: Full Signal object (not just ID) for context

    Returns:
        Signal ID if deposited successfully, None if rejected/failed
    """
    # Validate parent exists in store
    if parent_signal and parent_signal.id not in signal_store.signals:
        logger.warning(f"Parent {parent_signal.id} not found in store, depositing as orphan")
        parent_signal = None

    # Build provenance chain (full ancestry)
    provenance_chain = []
    if parent_signal:
        provenance_chain = parent_signal.get_provenance_chain() + [parent_signal.id]

    # Enhanced metadata for evidence chaining
    metadata = {
        "parent_content": parent_signal.content if parent_signal else None,
        "parent_type": parent_signal.type if parent_signal else None,
        "provenance_chain": provenance_chain,
        "depositor": depositor,
        "deposit_timestamp": time.time()
    }

    # Deposit with metadata
    signal_id = signal_store.deposit(
        signal_type, content, strength, depositor,
        parent=parent_signal.id if parent_signal else None,
        metadata=metadata
    )

    return signal_id


def validate_reference_quality(child_signal: Signal, parent_signal: Signal) -> float:
    """Validate how well child signal references its parent.

    Uses heuristic keyword overlap to determine reference quality.
    This ensures evidence chains maintain topical coherence.

    Args:
        child_signal: The child signal
        parent_signal: The parent signal

    Returns:
        Quality score 0.0-1.0:
        - 0.0-0.3: Poor reference (minimal overlap)
        - 0.3-0.6: Moderate reference (some overlap)
        - 0.6-1.0: Strong reference (high overlap)
    """
    if not child_signal or not parent_signal:
        return 0.0

    # Extract keywords (simple tokenization)
    parent_keywords = set(parent_signal.content.lower().split())
    child_keywords = set(child_signal.content.lower().split())

    # Remove stop words (very basic)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been'}
    parent_keywords = parent_keywords - stop_words
    child_keywords = child_keywords - stop_words

    if not parent_keywords:
        return 0.5  # Can't judge, assume moderate

    # Calculate overlap
    keyword_overlap = len(parent_keywords & child_keywords) / len(parent_keywords)

    # Map overlap to quality score
    if keyword_overlap < 0.1:
        return 0.0  # No meaningful reference
    elif keyword_overlap < 0.3:
        return 0.3  # Weak reference
    elif keyword_overlap < 0.5:
        return 0.6  # Moderate reference
    else:
        return min(1.0, 0.6 + (keyword_overlap - 0.5) * 0.8)  # Strong reference
