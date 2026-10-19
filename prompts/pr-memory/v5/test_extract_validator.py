import json
import re
from pathlib import Path

import pytest

from extract_validator import (
    FIELDS,
    OutputFormatError,
    period_end,
    user_turns,
    validate_output,
)

HERE = Path(__file__).parent
SESSION_DATE = "2026-10-19"
CONVERSATION = """\
User: Saw the dentist on Friday, no cavities. And my doctor switched me from lisinopril to losartan.
Assistant: Good news on the teeth. Any side effects from the new medication so far?
User: None. I want to get to bed by 11 on weeknights. I might try yoga at some point.
"""
PROFILE = [
    {"id": "p_3", "fact": "User is vegetarian", "pillar": "nutrition_food", "privacy_level": "LOW"},
    {"id": "p_7", "fact": "User takes lisinopril", "pillar": "physiological", "privacy_level": "LOW"},
    {"id": "p_9", "fact": "User drinks alcohol most nights", "pillar": "lifestyle", "privacy_level": "HIGH"},
]
EPISODIC = [
    {"id": "e_1", "fact": "User ran 5 km", "pillar": "activity", "privacy_level": "LOW",
     "anchor_date": "2026-10-12", "granularity": "day"},
]
INTENT = [
    {"id": "i_2", "fact": "User has a dentist appointment", "pillar": "physiological",
     "privacy_level": "LOW", "status": "expired", "due_date": "2026-10-16", "due_granularity": "day"},
    {"id": "i_4", "fact": "User wants to run a marathon", "pillar": "activity",
     "privacy_level": "LOW", "status": "active", "due_date": None, "due_granularity": None},
]


def action(**overrides):
    base = dict.fromkeys(FIELDS)
    base.update(action="NEW", memory_type="profile", fact="User takes losartan",
                evidence="my doctor switched me from lisinopril to losartan",
                pillar="physiological", privacy_level="LOW", user_explicit=True,
                memorability="high")
    base.update(overrides)
    return base


def run(*actions, conversation=CONVERSATION):
    return validate_output({"scratchpad": "", "actions": list(actions)},
                           session_date=SESSION_DATE, conversation=conversation,
                           profile_items=PROFILE, episodic_items=EPISODIC, intent_items=INTENT)


def errors_of(*actions, **kwargs):
    result = run(*actions, **kwargs)
    return [error for reject in result.rejects for error in reject.errors]


def assert_rejected(act, fragment, **kwargs):
    errors = errors_of(act, **kwargs)
    assert any(fragment in error for error in errors), errors


def test_prompt_example_is_valid():
    template = (HERE / "pr-template.jinja").read_text()
    example = template[template.index("Output:\n") + len("Output:\n"):template.index("</EXAMPLE>")]
    conversation = re.search(r"CONVERSATION_HISTORY:\n(.*?)\n\nOutput:", template, re.S).group(1)
    result = validate_output(example, session_date=SESSION_DATE, conversation=conversation,
                             profile_items=PROFILE, intent_items=INTENT)
    assert result.rejects == []
    assert [a["action"] for a in result.valid] == ["RESOLVED", "UPDATE", "NEW"]


def test_empty_actions_is_noop():
    result = run()
    assert result.valid == [] and result.rejects == []


@pytest.mark.parametrize("text", [
    "not json",
    "[]",
    '{"actions": []}',
    '{"scratchpad": "", "actions": [], "classifications": []}',
    '{"scratchpad": 1, "actions": []}',
])
def test_unusable_output_raises(text):
    with pytest.raises(OutputFormatError):
        validate_output(text, session_date=SESSION_DATE, conversation=CONVERSATION)


def test_missing_and_unknown_fields():
    act = action()
    del act["stability"]
    act["reasoning"] = "x"
    errors = errors_of(act)
    assert any("missing fields: stability" in e for e in errors)
    assert any("unknown fields: reasoning" in e for e in errors)


@pytest.mark.parametrize("field,value", [
    ("pillar", "stress_mood"),
    ("pillar", "medical"),
    ("action", "MERGE"),
    ("privacy_level", "low"),
])
def test_invalid_enum_values(field, value):
    assert_rejected(action(**{field: value}), f"{field} has invalid value")


def test_user_explicit_must_be_boolean():
    assert_rejected(action(user_explicit="true"), "user_explicit must be a boolean")


def test_new_rejects_target_id():
    assert_rejected(action(target_id="p_7"), "NEW requires target_id to be null")


def test_profile_must_not_carry_stability_or_dates():
    assert_rejected(action(stability="stable"), "requires stability to be null")
    assert_rejected(action(anchor_date="2026-10-01", granularity="month"),
                    "requires anchor_date to be null")


def test_intent_requires_stability_and_paired_due_fields():
    intent = action(memory_type="intent", fact="User wants to go to bed by 11 PM on weeknights",
                    evidence="I want to get to bed by 11 on weeknights.", pillar="sleep",
                    memorability="medium")
    assert_rejected(intent, "requires stability")
    assert_rejected(dict(intent, stability="moderate", due_date="2026-11-01"),
                    "due_date and due_granularity must both be set")
    assert errors_of(dict(intent, stability="moderate")) == []


def test_episodic_dates():
    event = action(memory_type="episodic", fact="User had a dental checkup",
                   evidence="Saw the dentist on Friday", memorability="low")
    assert_rejected(event, "requires anchor_date")
    assert errors_of(dict(event, anchor_date="2026-10-16", granularity="day")) == []
    assert_rejected(dict(event, anchor_date="2026-10-20", granularity="day"),
                    "must not be after session_date")
    assert_rejected(dict(event, anchor_date="2026-10-16", granularity="month"),
                    "first day of the month")
    assert_rejected(dict(event, anchor_date="2026-02-30", granularity="day"), "not a real date")


def test_due_date_compares_against_period_end():
    intent = action(memory_type="intent", fact="User wants to lose 5 kg", evidence="I want to get to bed",
                    pillar="activity", memorability="high", stability="stable")
    assert errors_of(dict(intent, due_date="2026-10-01", due_granularity="month")) == []
    assert_rejected(dict(intent, due_date="2026-10-18", due_granularity="day"),
                    "must not be before session_date")


def test_target_must_exist_and_match_type():
    assert_rejected(action(action="CONFIRM", target_id="p_404", fact=None, pillar=None,
                           privacy_level=None, memorability=None),
                    "is not an existing item")
    assert_rejected(action(action="CONFIRM", memory_type="intent", target_id="p_7", fact=None,
                           pillar=None, privacy_level=None, memorability=None),
                    "does not match target type")


def test_confirm_must_not_carry_attributes():
    assert_rejected(action(action="CONFIRM", target_id="p_7"), "CONFIRM requires fact to be null")


def test_update_rules():
    update = action(action="UPDATE", target_id="p_7", reason="change")
    assert errors_of(update) == []
    assert_rejected(dict(update, reason="no_longer_true"), "UPDATE requires reason")
    assert_rejected(dict(update, intent_status="paused"), "requires intent_status to be null")


def test_update_cannot_lower_privacy():
    assert_rejected(action(action="UPDATE", target_id="p_9", reason="change",
                           fact="User drinks alcohol on weekends", pillar="lifestyle",
                           memorability="medium", evidence="I might try yoga"),
                    "must not lower privacy_level")


def test_update_can_pause_intent():
    pause = action(action="UPDATE", memory_type="intent", target_id="i_4", reason="change",
                   intent_status="paused", fact="User wants to run a marathon", pillar="activity",
                   memorability="high", stability="stable", evidence="I might try yoga")
    assert errors_of(pause) == []


def test_remove_reasons_depend_on_type():
    remove = action(action="REMOVE", target_id="p_3", reason="no_longer_true", fact=None,
                    pillar=None, privacy_level=None, memorability=None, evidence="I might try yoga")
    assert errors_of(remove) == []
    assert_rejected(dict(remove, memory_type="intent", target_id="i_4"),
                    "REMOVE of intent requires reason")
    assert errors_of(dict(remove, memory_type="intent", target_id="i_4",
                          reason="user_requested")) == []


def test_resolved_and_abandoned_only_for_intent():
    abandoned = action(action="ABANDONED", memory_type="profile", target_id="p_3", fact=None,
                       pillar=None, privacy_level=None, memorability=None, evidence="I might try yoga")
    assert_rejected(abandoned, "ABANDONED applies only to intent")


def test_resolved_event_fields_all_or_nothing():
    resolved = action(action="RESOLVED", memory_type="intent", target_id="i_2", fact=None,
                      pillar=None, privacy_level=None, memorability=None,
                      evidence="Saw the dentist on Friday")
    assert errors_of(resolved) == []
    assert_rejected(dict(resolved, fact="User had a dental checkup"),
                    "RESOLVED must fill all event fields")


def test_double_write_is_two_valid_news():
    conversation = "User: I was diagnosed with type 2 diabetes in May.\n"
    evidence = "I was diagnosed with type 2 diabetes in May."
    episode = action(memory_type="episodic", fact="User was diagnosed with type 2 diabetes",
                     evidence=evidence, anchor_date="2026-05-01", granularity="month")
    profile = action(fact="User has type 2 diabetes", evidence=evidence)
    result = run(episode, profile, conversation=conversation)
    assert result.rejects == [] and len(result.valid) == 2


def test_evidence_must_come_from_user():
    assert_rejected(action(evidence="Any side effects from the new medication so far?"),
                    "not a quote from a User turn")


def test_evidence_handles_timestamps_and_wrapped_turns():
    conversation = ("[2026-10-19 08:12] User: My doctor switched me\n"
                    "from lisinopril to losartan.\n"
                    "[2026-10-19 08:13] Assistant: Noted.\n")
    assert user_turns(conversation) == ["My doctor switched me\nfrom lisinopril to losartan."]
    assert errors_of(action(evidence="switched me from lisinopril"), conversation=conversation) == []


def test_two_actions_on_one_target_are_both_rejected():
    confirm = action(action="CONFIRM", target_id="p_7", fact=None, pillar=None,
                     privacy_level=None, memorability=None)
    update = action(action="UPDATE", target_id="p_7", reason="change")
    result = run(confirm, update)
    assert result.valid == [] and len(result.rejects) == 2


def test_valid_actions_survive_alongside_rejects():
    result = run(action(), action(pillar="medical"))
    assert len(result.valid) == 1
    assert [r.index for r in result.rejects] == [1]


@pytest.mark.parametrize("granularity,expected", [
    ("day", "2028-02-10"),
    ("month", "2028-02-29"),
    ("year", "2028-12-31"),
])
def test_period_end(granularity, expected):
    from datetime import date
    assert period_end(date(2028, 2, 10), granularity).isoformat() == expected


def test_schema_and_validator_agree_on_fields():
    schema = json.loads((HERE / "output-schema.json").read_text())
    item = schema["properties"]["actions"]["items"]
    assert tuple(item["required"]) == FIELDS
    assert set(item["properties"]) == set(FIELDS)
