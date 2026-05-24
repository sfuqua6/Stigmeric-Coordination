"""Core stigmergic primitives.

Components:
    config         — module-level tunables (validated at import)
    signal_types   — universal signal type constants
    signal_store   — typed DAG with strength dynamics; trace-only API
    intake         — corpus partitioner (one partition per scout)
    sampling       — differentiated sampling strategies for downstream agents
    diversity      — Jaccard distance over agent context sets
    llm            — single LLM wrapper with semaphore + MockLLM
"""
