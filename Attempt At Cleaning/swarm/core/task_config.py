"""Adaptive task configuration system for the swarm.

The swarm uses a UNIFIED ADAPTIVE configuration that works for any task type.
There are no separate "modes" - the system adapts based on the task prompt itself.

IMPORTANT: All tasks use UNIVERSAL signal types (INITIAL, SUPPORT, CRITIQUE, OBJECTION).
The system automatically adjusts its behavior based on the task prompt content.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from .signal_types import SignalType, INITIAL, SUPPORT, CRITIQUE, OBJECTION


@dataclass
class IntakeProfile:
    """Knowledge intake configuration for research depth and quality.

    Intake profiles control how much research the swarm performs:
    - CREATIVE_INTAKE: Light research, high diversity, fast iteration
    - DEFAULT_INTAKE: Balanced research for general use
    - TECHNICAL_INTAKE: Deep research, authoritative sources
    - DEBATE_INTAKE: Balanced pro/con coverage, high fact-checking
    - ADAPTIVE_INTAKE: Maximum quality - extensive research, high token limits

    The swarm adapts to any task type regardless of intake profile.
    """

    # Research volume
    target_words_per_round: int = 100000      # Target words to ingest (30K - 500K)
    max_sources_per_keyword: int = 5          # Max sources to fetch per keyword (3 - 15)
    research_rounds: int = 1                  # Number of research rounds (1 - 3)

    # Content processing
    chunk_size: int = 500                     # Words per chunk (200 - 800)
    chunk_overlap: int = 60                   # Overlap between chunks (30 - 100)

    # Token allocation (scout token allocation)
    scout_tokens: int = 150                   # Tokens per scout insight (50 - 250)
    forager_tokens: int = 100                 # Tokens per forager development (80 - 150)
    critic_tokens: int = 150                  # Tokens per critic evaluation (100 - 200)
    hater_tokens: int = 130                   # Tokens per hater objection (100 - 180)
    synthesizer_tokens: int = 300             # Tokens for final synthesis (150 - 400)

    # Fragment management
    fragment_assignment: str = "round_robin"  # "round_robin", "clustered", "balanced_procon"
    prioritization: str = "importance"        # "importance", "diversity", "coverage", "quality"
    max_fragments_per_scout: int = 100        # Limit fragments per scout (10 - 200)

    # Quality thresholds
    min_fragment_quality: float = 0.4         # Minimum quality score (0.3 - 0.8)
    fact_check_threshold: float = 0.7         # Threshold for fact-checking rigor (0.5 - 0.9)

    # Source preferences
    source_types: List[str] = field(default_factory=lambda: ["wikipedia", "duckduckgo"])
    source_credibility_weight: float = 0.6    # Weight for source credibility (0.0 - 1.0)


@dataclass
class TaskConfig:
    """Unified adaptive configuration for any task type.

    The swarm uses task-agnostic prompts and automatically adapts
    its behavior based on the task_prompt content.
    """
    task_type: str  # Type identifier (now always "adaptive")
    task_prompt: str  # The user's task, question, or problem
    signal_types: Dict[str, str]  # Maps roles to UNIVERSAL signal types (INITIAL, SUPPORT, CRITIQUE, OBJECTION)
    display_names: Dict[str, str] = field(default_factory=dict)  # Generic labels for output (Insight, Support, etc.)
    intake_profile: IntakeProfile = field(default_factory=IntakeProfile)  # Research depth configuration
    scout_prompt_template: str = ""
    forager_evidence_prompt_template: str = ""
    forager_critique_prompt_template: str = ""
    critic_prompt_template: str = ""
    hater_prompt_template: str = ""
    # Set by Planner before the swarm runs; None = use legacy template behavior
    task_frame: object = None  # TaskFrame | None


# ============================================================================
# INTAKE PROFILES - Task-Adaptive Knowledge Intake Configurations
# ============================================================================

# CREATIVE INTAKE: Light research, high diversity, relaxed fact-checking
CREATIVE_INTAKE = IntakeProfile(
    target_words_per_round=50000,        # Light research (50K words)
    max_sources_per_keyword=6,           # Breadth over depth
    research_rounds=1,
    chunk_size=400,                      # Medium chunks
    chunk_overlap=50,
    scout_tokens=150,                    # Standard tokens
    forager_tokens=100,
    critic_tokens=130,
    hater_tokens=120,
    synthesizer_tokens=350,              # High for creative synthesis
    fragment_assignment="round_robin",
    prioritization="diversity",          # Favor unique/novel findings
    max_fragments_per_scout=60,
    min_fragment_quality=0.3,            # Relaxed quality
    fact_check_threshold=0.5,            # Relaxed fact-checking
    source_types=["wikipedia", "duckduckgo"],
    source_credibility_weight=0.4
)

# TECHNICAL/ANALYSIS INTAKE: Deep research, authoritative sources, rigorous verification
TECHNICAL_INTAKE = IntakeProfile(
    target_words_per_round=300000,       # Deep research (300K words)
    max_sources_per_keyword=10,          # Depth + verification
    research_rounds=2,                   # Multi-round for refinement
    chunk_size=600,                      # Large chunks preserve technical context
    chunk_overlap=80,
    scout_tokens=200,                    # High for technical detail
    forager_tokens=120,
    critic_tokens=180,
    hater_tokens=150,
    synthesizer_tokens=250,              # Medium (good signals = less synthesis)
    fragment_assignment="clustered",     # Thematic coherence
    prioritization="quality",            # Favor authoritative sources
    max_fragments_per_scout=120,
    min_fragment_quality=0.6,            # Higher quality threshold
    fact_check_threshold=0.8,            # Rigorous verification
    source_types=["wikipedia", "duckduckgo"],  # TODO: Add arxiv, scholar
    source_credibility_weight=0.8
)

# DEBATE INTAKE: Balanced research, pro/con coverage, high fact-checking
DEBATE_INTAKE = IntakeProfile(
    target_words_per_round=250000,       # Balanced depth (250K words)
    max_sources_per_keyword=12,          # Pro + con sources
    research_rounds=2,
    chunk_size=500,                      # Standard chunks
    chunk_overlap=60,
    scout_tokens=180,                    # High for argument detail
    forager_tokens=110,
    critic_tokens=160,
    hater_tokens=140,
    synthesizer_tokens=250,
    fragment_assignment="round_robin",   # TODO: Implement balanced_procon
    prioritization="importance",         # Balance important claims from both sides
    max_fragments_per_scout=100,
    min_fragment_quality=0.5,
    fact_check_threshold=0.75,           # High (avoid misinformation in debates)
    source_types=["wikipedia", "duckduckgo"],
    source_credibility_weight=0.7
)

# PROBLEM SOLVING INTAKE: Moderate research, solution-oriented, practical focus
PROBLEM_SOLVING_INTAKE = IntakeProfile(
    target_words_per_round=150000,       # Moderate research (150K words)
    max_sources_per_keyword=8,
    research_rounds=1,
    chunk_size=500,
    chunk_overlap=60,
    scout_tokens=160,
    forager_tokens=110,
    critic_tokens=150,
    hater_tokens=130,
    synthesizer_tokens=280,
    fragment_assignment="round_robin",
    prioritization="importance",
    max_fragments_per_scout=80,
    min_fragment_quality=0.5,
    fact_check_threshold=0.7,
    source_types=["wikipedia", "duckduckgo"],
    source_credibility_weight=0.6
)

# DEFAULT INTAKE: Balanced configuration for unknown task types
DEFAULT_INTAKE = IntakeProfile(
    target_words_per_round=100000,       # Moderate research (100K words)
    max_sources_per_keyword=5,
    research_rounds=1,
    chunk_size=500,
    chunk_overlap=60,
    scout_tokens=150,
    forager_tokens=100,
    critic_tokens=150,
    hater_tokens=130,
    synthesizer_tokens=300,
    fragment_assignment="round_robin",
    prioritization="importance",
    max_fragments_per_scout=100,
    min_fragment_quality=0.4,
    fact_check_threshold=0.7,
    source_types=["wikipedia", "duckduckgo"],
    source_credibility_weight=0.6
)

# ADAPTIVE INTAKE: High-quality configuration that adapts to any task type
# The swarm automatically adjusts behavior based on the task prompt itself
ADAPTIVE_INTAKE = IntakeProfile(
    target_words_per_round=400000,       # Extensive research (400K words)
    max_sources_per_keyword=15,          # Maximum breadth and depth
    research_rounds=2,                   # Multi-round refinement
    chunk_size=700,                      # Large chunks for context preservation
    chunk_overlap=100,                   # High overlap for continuity
    scout_tokens=350,                    # High - allow detailed, well-developed insights
    forager_tokens=220,                  # High - thorough evidence and development
    critic_tokens=280,                   # High - comprehensive, nuanced critiques
    hater_tokens=250,                    # High - well-argued counterpoints
    synthesizer_tokens=600,              # Very high - detailed, well-structured synthesis
    fragment_assignment="clustered",     # Thematic coherence
    prioritization="quality",            # Favor high-quality content
    max_fragments_per_scout=150,         # More fragments for comprehensive coverage
    min_fragment_quality=0.6,            # Higher quality threshold
    fact_check_threshold=0.8,            # Rigorous verification
    source_types=["wikipedia", "duckduckgo"],
    source_credibility_weight=0.8        # Emphasize credible sources
)

# Alias for backward compatibility
HIGH_QUALITY_INTAKE = ADAPTIVE_INTAKE


# ============================================================================
# ADAPTIVE TASK CONFIG - Unified configuration for all task types
# ============================================================================

ADAPTIVE_CONFIG = TaskConfig(
    task_type="adaptive",
    task_prompt="",  # Set by user when creating task
    signal_types={
        "initial": INITIAL,
        "support": SUPPORT,
        "critique": CRITIQUE,
        "counter": OBJECTION
    },
    display_names={
        INITIAL: "Insight",
        SUPPORT: "Support",
        CRITIQUE: "Critique",
        OBJECTION: "Challenge"
    },
    intake_profile=ADAPTIVE_INTAKE,  # High-quality adaptive intake

    scout_prompt_template="""You are exploring this task:
"{task_prompt}"

Generate a clear, specific, and well-developed idea related to this task.

Quality guidelines:
- Be specific and concrete (avoid vague generalizations)
- Include context or reasoning (explain why this matters)
- Make meaningful contributions (avoid obvious or trivial points)
- Consider implications, consequences, or deeper connections
- Aim for insight and depth

Your contribution (2-4 sentences):""",

    forager_evidence_prompt_template="""Developing this idea:
"{parent_content}"

Task context: "{task_prompt}"

Provide detailed support, evidence, or elaboration for this idea.

Quality guidelines:
- Be specific and concrete (data, examples, research findings)
- Explain HOW your contribution connects to or supports the idea
- Add substantive depth (don't just restate)
- Include sources, details, or mechanisms when relevant
- Consider multiple angles of support
- Acknowledge relevant limitations or boundaries

Your supporting contribution (2-4 sentences):""",

    forager_critique_prompt_template="""Analyzing this idea:
"{parent_content}"

Task context: "{task_prompt}"

Identify specific weaknesses, limitations, gaps, or counterarguments.

Quality guidelines:
- Point to concrete issues (not vague criticisms)
- Explain WHY each concern matters
- Be constructive (suggest what would strengthen the idea)
- Consider alternative interpretations or edge cases
- Distinguish major flaws from minor limitations

Your analytical critique (2-4 sentences):""",

    critic_prompt_template="""Evaluating this contribution to the task:
"{task_prompt}"

Specific idea being evaluated:
"{parent_content}"

Provide a thorough analytical evaluation of this idea.

Quality guidelines:
- Identify specific strengths and weaknesses
- Point out logical gaps, unsupported assumptions, or missing evidence
- Consider alternative perspectives or counterarguments
- Explain WHY identified issues matter to the overall task
- Be constructive - suggest improvements
- Assess both validity and usefulness

Your evaluation (2-4 sentences):""",

    hater_prompt_template="""Challenging this contribution:
"{parent_content}"

Task context: "{task_prompt}"

Generate a substantive counterargument, alternative perspective, or contradictory evidence.

Quality guidelines:
- Be specific and concrete (not just "I disagree")
- Cite alternative evidence, perspectives, or reasoning
- Explain WHY this challenges the original contribution
- Consider what the original perspective misses or overlooks
- Aim for intellectual honesty and rigor

Your challenge (2-4 sentences):"""
)


# ============================================================================
# DEPRECATED: Legacy Task-Specific Configs
#
# These are maintained for backward compatibility only.
# All configs now point to ADAPTIVE_CONFIG, which adapts to any task type.
# ============================================================================

# All legacy configs are just aliases to ADAPTIVE_CONFIG
DEBATE_CONFIG = ADAPTIVE_CONFIG
CREATIVE_CONFIG = ADAPTIVE_CONFIG
ANALYSIS_CONFIG = ADAPTIVE_CONFIG
PROBLEM_SOLVING_CONFIG = ADAPTIVE_CONFIG


# ============================================================================
# Task Registry and Selection
# ============================================================================

# Legacy task registry - all task types now use ADAPTIVE_CONFIG
# Maintained for backward compatibility with existing code
TASK_CONFIGS = {
    "adaptive": ADAPTIVE_CONFIG,
    "debate": ADAPTIVE_CONFIG,
    "creative": ADAPTIVE_CONFIG,
    "analysis": ADAPTIVE_CONFIG,
    "problem_solving": ADAPTIVE_CONFIG
}


def get_task_config(task_type: str = "adaptive") -> TaskConfig:
    """Get configuration for a task type.

    All task types now return ADAPTIVE_CONFIG, which adapts to any task.
    The task_type parameter is maintained for backward compatibility.

    Args:
        task_type: Legacy parameter (ignored). Defaults to "adaptive".

    Returns:
        ADAPTIVE_CONFIG that works for any task type
    """
    # All task types now use the adaptive config
    return ADAPTIVE_CONFIG


def create_custom_task(task_type_or_prompt: str, custom_prompt: str = None, intake_profile: IntakeProfile = None) -> TaskConfig:
    """Create a task config with a custom prompt.

    The system automatically adapts to any task type based on the prompt.

    Supports both old and new signatures for backward compatibility:
    - NEW: create_custom_task("Your question here")
    - OLD: create_custom_task("debate", "Your question here")

    Args:
        task_type_or_prompt: Either your task prompt (new style) or task type (old style, ignored)
        custom_prompt: The prompt if using old style (deprecated)
        intake_profile: Optional custom intake profile (defaults to ADAPTIVE_INTAKE)

    Returns:
        TaskConfig with custom prompt and adaptive behavior

    Examples:
        task = create_custom_task("Should we regulate AI development?")
        task = create_custom_task("Write a poem about hope", intake_profile=CREATIVE_INTAKE)
        task = create_custom_task("debate", "Old style call")  # Still works
    """
    from copy import deepcopy

    config = deepcopy(ADAPTIVE_CONFIG)

    # Detect old vs new signature
    if custom_prompt is not None:
        # Old signature: create_custom_task(task_type, custom_prompt)
        # Ignore task_type (first arg), use custom_prompt
        config.task_prompt = custom_prompt
    else:
        # New signature: create_custom_task(task_prompt)
        config.task_prompt = task_type_or_prompt

    if intake_profile is not None:
        config.intake_profile = intake_profile

    return config
