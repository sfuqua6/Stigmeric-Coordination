r"""extract_json_object + unified validator parsing.

The old extraction regex r"\{[^{}]*\}" matched only FLAT objects: any brace
inside the reasoning string, or a reasoning-model preamble containing {...},
made it fail or capture garbage — the validator then silently defaulted to
score 0.5 (the failures behind validator_raw.log). These tests pin the
brace-balanced replacement and the legacy Validator's delegation to the
task-aware schema in core.actions.
"""
from core.actions import extract_json_object, validate_parse


def test_flat_object():
    assert extract_json_object('{"supports": true, "confidence": 0.9}') == {
        "supports": True, "confidence": 0.9}


def test_nested_brace_in_string():
    # The exact case the flat regex could not match.
    raw = '{"supports": true, "confidence": 0.8, "reasoning": "the {source} confirms it"}'
    obj = extract_json_object(raw)
    assert obj is not None and obj["confidence"] == 0.8


def test_preamble_with_stray_braces_before_real_object():
    raw = ('Let me think {step by step... unbalanced\n'
           'Answer: {"engages": true, "quality": 0.7, "reasoning": "on-topic"}')
    obj = extract_json_object(raw)
    assert obj == {"engages": True, "quality": 0.7, "reasoning": "on-topic"}


def test_escaped_quotes_inside_string():
    raw = '{"reasoning": "she said \\"no\\"", "supports": false, "confidence": 0.6}'
    obj = extract_json_object(raw)
    assert obj is not None and obj["supports"] is False


def test_no_object_returns_none():
    assert extract_json_object("no json here { just a brace") is None
    assert extract_json_object("") is None


def test_non_dict_json_skipped():
    # A balanced block that parses to a dict later in the text is still found.
    raw = 'scores: {broken {\n then {"supports": true, "confidence": 1.0}'
    obj = extract_json_object(raw)
    assert obj == {"supports": True, "confidence": 1.0}


def test_validate_parse_nested_json_factual():
    # Under the old regex this fell through to the 0.5 default.
    raw = '{"supports": true, "confidence": 0.9, "reasoning": "matches {2020} data"}'
    pd = validate_parse(raw, task_type="coding")
    assert pd.strength == 0.9


def test_validate_parse_nonfactual_engages():
    raw = 'Sure! {"engages": true, "quality": 0.7, "reasoning": "substantive"}'
    pd = validate_parse(raw, task_type="debate")
    assert pd.strength == 0.7


def test_legacy_validator_delegates_schema():
    # The legacy class previously carried its own flat-regex parser with a
    # diverged schema ("note" key, factual-only). It must now match
    # validate_parse exactly, including the non-factual branch.
    from agents.validator import Validator
    v = Validator.__new__(Validator)  # skip __init__ (no LLM needed)
    v.agent_id = "t"
    v._task_type = "debate"
    raw = '{"engages": true, "quality": 0.8, "reasoning": "engages {topic}"}'
    note, score = v.parse(raw)
    assert score == 0.8
    assert "engages {topic}" in note
