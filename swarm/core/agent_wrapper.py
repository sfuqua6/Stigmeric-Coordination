"""Robust wrapper for agent execution with error handling and monitoring.

This module provides infrastructure for running agents concurrently with:
- Graceful failure handling (one agent failing doesn't crash the swarm)
- Health monitoring and statistics
- Automatic retry logic
- Detailed error logging
- Resource cleanup
"""

import asyncio
import traceback
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class AgentStats:
    """Statistics for agent performance."""
    agent_id: str
    start_time: float
    end_time: Optional[float] = None
    actions_completed: int = 0
    errors_encountered: int = 0
    last_error: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed, stopped

    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class SwarmHealth:
    """Overall swarm health metrics."""
    total_agents: int = 0
    running_agents: int = 0
    completed_agents: int = 0
    failed_agents: int = 0
    stopped_agents: int = 0
    total_actions: int = 0
    total_errors: int = 0
    start_time: float = field(default_factory=time.time)

    def health_score(self) -> float:
        """Calculate health score 0.0-1.0."""
        if self.total_agents == 0:
            return 1.0

        # Success rate based on completed vs failed
        completed_rate = self.completed_agents / self.total_agents
        failure_rate = self.failed_agents / self.total_agents

        # Penalize failures, reward completions
        score = completed_rate - (failure_rate * 2.0)

        # If too many failures (>50%), system is unhealthy
        if failure_rate > 0.5:
            return 0.0

        return max(0.0, min(1.0, score))

    def is_healthy(self) -> bool:
        """Check if swarm is healthy (>10% success rate, <50% failure rate)."""
        if self.total_agents == 0:
            return True

        failure_rate = self.failed_agents / self.total_agents
        success_rate = self.completed_agents / self.total_agents

        return failure_rate < 0.5 and (success_rate > 0.1 or self.completed_agents > 0)


class AgentExecutionWrapper:
    """Wrapper for safe concurrent agent execution."""

    def __init__(self, enable_retry: bool = True, max_retries: int = 2):
        """Initialize wrapper.

        Args:
            enable_retry: Enable automatic retry on failure
            max_retries: Maximum retry attempts per agent
        """
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.agent_stats: Dict[str, AgentStats] = {}
        self.swarm_health = SwarmHealth()

    async def run_agent_safe(self, agent: Any, *args, **kwargs) -> Optional[Any]:
        """Run agent with error handling and retry logic.

        Args:
            agent: Agent instance with run() method
            *args, **kwargs: Arguments to pass to agent.run()

        Returns:
            Result from agent.run() or None if failed
        """
        agent_id = getattr(agent, 'agent_id', str(agent))

        # Initialize stats
        stats = AgentStats(
            agent_id=agent_id,
            start_time=time.time(),
            status="running"
        )
        self.agent_stats[agent_id] = stats
        self.swarm_health.total_agents += 1
        self.swarm_health.running_agents += 1

        result = None
        attempt = 0

        while attempt <= self.max_retries:
            try:
                # Run agent
                result = await agent.run(*args, **kwargs)

                # Success
                stats.status = "completed"
                stats.end_time = time.time()
                stats.actions_completed = getattr(agent, 'actions_taken', 0)

                self.swarm_health.running_agents -= 1
                self.swarm_health.completed_agents += 1
                self.swarm_health.total_actions += stats.actions_completed

                print(f"[AGENT_WRAPPER] {agent_id} completed successfully "
                      f"({stats.actions_completed} actions, {stats.duration():.1f}s)")

                break

            except Exception as e:
                attempt += 1
                stats.errors_encountered += 1
                stats.last_error = f"{type(e).__name__}: {str(e)}"
                self.swarm_health.total_errors += 1

                error_msg = f"[AGENT_WRAPPER] {agent_id} error (attempt {attempt}/{self.max_retries + 1}): {stats.last_error}"

                if attempt <= self.max_retries and self.enable_retry:
                    print(f"{error_msg} - RETRYING")
                    await asyncio.sleep(1.0 * attempt)  # Exponential backoff
                else:
                    # Final failure
                    print(f"{error_msg} - FAILED")
                    print(f"[AGENT_WRAPPER] Traceback:\n{traceback.format_exc()}")

                    stats.status = "failed"
                    stats.end_time = time.time()

                    self.swarm_health.running_agents -= 1
                    self.swarm_health.failed_agents += 1

                    break

        return result

    async def run_agent_swarm(self, agents: list, *args, **kwargs) -> list:
        """Run multiple agents concurrently with isolation.

        Each agent runs independently - failures don't crash other agents.

        Args:
            agents: List of agent instances
            *args, **kwargs: Arguments to pass to each agent.run()

        Returns:
            List of results (None for failed agents)
        """
        print(f"[AGENT_WRAPPER] Launching swarm of {len(agents)} agents")

        # Create tasks for all agents
        tasks = [
            self.run_agent_safe(agent, *args, **kwargs)
            for agent in agents
        ]

        # Run all tasks, collecting results even if some fail
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to None
        results = [
            None if isinstance(r, Exception) else r
            for r in results
        ]

        # Report health
        health_score = self.swarm_health.health_score()
        print(f"\n[AGENT_WRAPPER] Swarm execution complete:")
        print(f"  Total agents: {self.swarm_health.total_agents}")
        print(f"  Completed: {self.swarm_health.completed_agents}")
        print(f"  Failed: {self.swarm_health.failed_agents}")
        print(f"  Total actions: {self.swarm_health.total_actions}")
        print(f"  Total errors: {self.swarm_health.total_errors}")
        print(f"  Health score: {health_score:.2f}")

        if not self.swarm_health.is_healthy():
            print(f"[AGENT_WRAPPER] WARNING: Swarm unhealthy (>50% failure rate)")

        return results

    def get_stats(self) -> Dict[str, AgentStats]:
        """Get stats for all agents."""
        return self.agent_stats

    def get_health(self) -> SwarmHealth:
        """Get overall swarm health."""
        return self.swarm_health

    def print_summary(self):
        """Print detailed execution summary."""
        print("\n" + "=" * 80)
        print("AGENT EXECUTION SUMMARY")
        print("=" * 80)

        # Group by status
        by_status = {}
        for stats in self.agent_stats.values():
            if stats.status not in by_status:
                by_status[stats.status] = []
            by_status[stats.status].append(stats)

        for status, agent_list in sorted(by_status.items()):
            print(f"\n{status.upper()} ({len(agent_list)}):")
            for stats in sorted(agent_list, key=lambda s: s.duration(), reverse=True)[:10]:
                print(f"  {stats.agent_id}: {stats.actions_completed} actions, "
                      f"{stats.duration():.1f}s, {stats.errors_encountered} errors")
                if stats.last_error:
                    print(f"    Last error: {stats.last_error}")

        print(f"\n{'=' * 80}")
        print(f"Health Score: {self.swarm_health.health_score():.2f}")
        print(f"{'=' * 80}\n")


# Convenience function for backward compatibility
async def run_agents_robust(agents: list, *args, **kwargs) -> list:
    """Run agents with robust error handling (convenience function).

    Args:
        agents: List of agents
        *args, **kwargs: Arguments for agent.run()

    Returns:
        List of results
    """
    wrapper = AgentExecutionWrapper()
    results = await wrapper.run_agent_swarm(agents, *args, **kwargs)
    wrapper.print_summary()
    return results
