"""Public interface for the junk/scratchpad content filter.

Thin re-export so external callers can import `is_junk` from a stable name
without depending on the internal `filters` module name.
"""

from .filters import is_junk_output as is_junk

__all__ = ["is_junk"]
