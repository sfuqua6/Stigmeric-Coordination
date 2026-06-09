"""Validation module for format and quality checking.

Includes:
- FormatValidator: Basic format checking
- DynamicKnowledgeBase: Learning knowledge base (Phase 4)
- External sources: Wikipedia, web search, symbolic math
- RealValidator: Actual fact-checking using external sources
"""

from .format_validator import FormatValidator

try:
    from .dynamic_knowledge_base import DynamicKnowledgeBase
    from .external_sources import WikipediaSource, WebSearchSource, SymbolicMathSource
    from .real_validator import RealValidator
    __all__ = [
        'FormatValidator',
        'DynamicKnowledgeBase',
        'WikipediaSource',
        'WebSearchSource',
        'SymbolicMathSource',
        'RealValidator',
    ]
except ImportError:
    # Phase 4 components not yet implemented
    __all__ = ['FormatValidator']
