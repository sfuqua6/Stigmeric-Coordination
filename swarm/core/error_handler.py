"""Centralized error handling for swarm agents."""

import time
import traceback
from typing import Optional, Dict, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Degraded but operational
    ERROR = "error"         # Operation failed, retryable
    CRITICAL = "critical"   # System-level failure, should stop


class SwarmError:
    """Structured error record."""

    def __init__(self, agent_id: str, error: Exception, context: Dict,
                 severity: ErrorSeverity = ErrorSeverity.ERROR):
        self.agent_id = agent_id
        self.error = error
        self.error_type = type(error).__name__
        self.message = str(error)
        self.context = context
        self.severity = severity
        self.timestamp = time.time()
        self.traceback = traceback.format_exc() if severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL] else None

    def __repr__(self):
        return (f"SwarmError({self.agent_id}, {self.error_type}, "
                f"{self.severity.value}, {self.context})")


class SwarmErrorHandler:
    """Centralized error handler for swarm operations.

    Benefits:
    - Consistent error logging across all agents
    - Error aggregation and statistics
    - Severity-based handling (graceful degradation)
    - Debugging support with context preservation
    """

    def __init__(self, verbose: bool = True):
        """Initialize error handler.

        Args:
            verbose: If True, print errors to console
        """
        self.verbose = verbose
        self.errors: List[SwarmError] = []
        self.error_counts: Dict[str, int] = {}  # error_type -> count

    def handle(self, agent_id: str, error: Exception, context: Dict,
               severity: ErrorSeverity = ErrorSeverity.ERROR) -> Optional[any]:
        """Handle an error from an agent.

        Args:
            agent_id: ID of agent that encountered error
            error: The exception that was raised
            context: Additional context (operation, signal_id, etc.)
            severity: Error severity level

        Returns:
            None for graceful degradation, or raises for critical errors
        """
        # Record error
        swarm_error = SwarmError(agent_id, error, context, severity)
        self.errors.append(swarm_error)

        # Update counts
        error_type = swarm_error.error_type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Handle based on severity
        if severity == ErrorSeverity.CRITICAL:
            if self.verbose:
                print(f"\n❌ CRITICAL ERROR [{agent_id}]: {error}")
                print(f"   Context: {context}")
                if swarm_error.traceback:
                    print(f"   {swarm_error.traceback}")
            raise error  # Re-raise critical errors

        elif severity == ErrorSeverity.ERROR:
            if self.verbose:
                print(f"⚠️  ERROR [{agent_id}]: {error_type}: {swarm_error.message}")
                if context:
                    print(f"   Context: {context}")
            return None  # Graceful degradation

        elif severity == ErrorSeverity.WARNING:
            if self.verbose:
                print(f"⚡ WARNING [{agent_id}]: {swarm_error.message}")
            return None

        else:  # INFO
            if self.verbose:
                print(f"ℹ️  INFO [{agent_id}]: {swarm_error.message}")
            return None

    def get_statistics(self) -> Dict:
        """Get error statistics.

        Returns:
            Dictionary with error metrics
        """
        total_errors = len(self.errors)
        by_severity = {}
        by_agent = {}

        for error in self.errors:
            # Count by severity
            sev = error.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

            # Count by agent
            by_agent[error.agent_id] = by_agent.get(error.agent_id, 0) + 1

        return {
            'total_errors': total_errors,
            'by_severity': by_severity,
            'by_agent': by_agent,
            'by_type': self.error_counts,
            'unique_error_types': len(self.error_counts)
        }

    def print_summary(self):
        """Print error summary."""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("ERROR SUMMARY")
        print("=" * 70)
        print(f"Total errors: {stats['total_errors']}")

        if stats['total_errors'] > 0:
            print(f"\nBy severity:")
            for severity, count in stats['by_severity'].items():
                print(f"  {severity}: {count}")

            print(f"\nBy error type (top 5):")
            sorted_types = sorted(stats['by_type'].items(),
                                 key=lambda x: x[1], reverse=True)
            for error_type, count in sorted_types[:5]:
                print(f"  {error_type}: {count}")

            if len(stats['by_agent']) > 0:
                print(f"\nBy agent (agents with errors):")
                sorted_agents = sorted(stats['by_agent'].items(),
                                     key=lambda x: x[1], reverse=True)
                for agent, count in sorted_agents[:10]:
                    print(f"  {agent}: {count}")

        print("=" * 70)

    def clear(self):
        """Clear all recorded errors."""
        self.errors.clear()
        self.error_counts.clear()


# Global error handler instance
_global_error_handler: Optional[SwarmErrorHandler] = None


def get_error_handler(verbose: bool = True) -> SwarmErrorHandler:
    """Get or create the global error handler.

    Args:
        verbose: If creating new handler, set verbosity

    Returns:
        Global SwarmErrorHandler instance
    """
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = SwarmErrorHandler(verbose=verbose)
    return _global_error_handler


def reset_error_handler():
    """Reset the global error handler."""
    global _global_error_handler
    _global_error_handler = None
