"""Agents.

Each agent role is a thin specialization of BaseAgent. Agents communicate
exclusively through the SignalStore — they never read each other's
prompts, generations, or reasoning chains.
"""
