"""Validation layer (S2-5) for pr-memory v5 extractor output.

Checks everything output-schema.json covers plus the cross-field rules it
cannot express. Invalid actions are returned as rejects for `extract_rejects`
and must not reach APPLY. Standard library only.
"""

from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping

FIELDS = (
    "action", "memory_type", "target_id", "fact", "evidence", "reason",
    "intent_status", "pillar", "privacy_level", "user_explicit", "memorability",
    "stability", "anchor_date", "granularity", "due_date", "due_granularity",
)
ACTIONS = {"NEW", "CONFIRM", "UPDATE", "REMOVE", "RESOLVED", "ABANDONED"}
MEMORY_TYPES = {"profile", "episodic", "intent"}
PILLARS = {"energy", "sleep", "activity", "nutrition_food", "stress",
           "physiological", "lifestyle", "other"}
PRIVACY_LEVELS = {"HIGH", "LOW"}
MEMORABILITY = {"high", "medium", "low"}
STABILITY = {"stable", "moderate", "dynamic"}
GRANULARITIES = {"day", "month", "year"}
INTENT_STATUSES = {"active", "paused"}
UPDATE_REASONS = {"change", "correction"}
REMOVE_REASONS = {"no_longer_true", "user_requested", "incorrect"}
INTENT_REMOVE_REASONS = {"user_requested", "incorrect"}

ENUMS = {
    "action": ACTIONS,
    "memory_type": MEMORY_TYPES,
    "reason": UPDATE_REASONS | REMOVE_REASONS,
    "intent_status": INTENT_STATUSES,
    "pillar": PILLARS,
    "privacy_level": PRIVACY_LEVELS,
    "memorability": MEMORABILITY,
    "stability": STABILITY,
    "granularity": GRANULARITIES,
    "due_granularity": GRANULARITIES,
}
ATTRIBUTES = ("fact", "pillar", "privacy_level", "memorability")
EVENT_FIELDS = ("fact", "pillar", "privacy_level", "memorability", "anchor_date", "granularity")
OPTIONAL_FIELDS = ("fact", "reason", "intent_status", "pillar", "privacy_level", "memorability",
                   "stability", "anchor_date", "granularity", "due_date", "due_granularity")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TURN_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?(User|Assistant)\s*:\s?(.*)$")


class OutputFormatError(ValueError):
    """The output as a whole cannot be used (bad JSON or wrong top-level shape)."""


@dataclass
class Reject:
    index: int
    action: Any
    errors: list[str]


@dataclass
class ValidationResult:
    valid: list[dict] = field(default_factory=list)
    rejects: list[Reject] = field(default_factory=list)


def parse_output(text: str) -> dict:
    try:
        output = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OutputFormatError(f"not valid JSON: {exc}") from exc
    if not isinstance(output, dict):
        raise OutputFormatError("top level must be a JSON object")
    if set(output) != {"scratchpad", "actions"}:
        raise OutputFormatError("top level must have exactly the keys scratchpad and actions")
    if not isinstance(output["scratchpad"], str):
        raise OutputFormatError("scratchpad must be a string")
    if not isinstance(output["actions"], list):
        raise OutputFormatError("actions must be an array")
    return output


def user_turns(conversation: str) -> list[str]:
    """Split a transcript into the text of its User turns."""
    turns: list[str] = []
    speaker = None
    for line in conversation.splitlines():
        match = _TURN_RE.match(line)
        if match:
            speaker = match.group(1)
            if speaker == "User":
                turns.append(match.group(2))
        elif speaker == "User":
            turns[-1] += "\n" + line
    return turns


def period_end(day: date, granularity: str) -> date:
    if granularity == "year":
        return date(day.year, 12, 31)
    if granularity == "month":
        return date(day.year, day.month, calendar.monthrange(day.year, day.month)[1])
    return day


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _parse_date(value: str, granularity: str | None, name: str, errors: list[str]) -> date | None:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        errors.append(f"{name} must be YYYY-MM-DD")
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        errors.append(f"{name} is not a real date")
        return None
    if granularity == "month" and parsed.day != 1:
        errors.append(f"{name} with month granularity must be the first day of the month")
    if granularity == "year" and (parsed.month, parsed.day) != (1, 1):
        errors.append(f"{name} with year granularity must be January 1")
    return parsed


def _check_shape(action: Mapping, errors: list[str]) -> None:
    missing = [name for name in FIELDS if name not in action]
    extra = [name for name in action if name not in FIELDS]
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"unknown fields: {', '.join(extra)}")
    for name, allowed in ENUMS.items():
        value = action.get(name)
        if value is not None and value not in allowed:
            errors.append(f"{name} has invalid value {value!r}")
    for name in ("target_id", "fact"):
        value = action.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{name} must be a non-empty string or null")
    if not isinstance(action.get("evidence"), str) or not action.get("evidence", "").strip():
        errors.append("evidence must be a non-empty string")
    if not isinstance(action.get("user_explicit"), bool):
        errors.append("user_explicit must be a boolean")
    if action.get("action") is None:
        errors.append("action is required")
    if action.get("memory_type") is None:
        errors.append("memory_type is required")


def _require(action: Mapping, names: Iterable[str], errors: list[str], context: str) -> None:
    for name in names:
        if action.get(name) is None:
            errors.append(f"{context} requires {name}")


def _forbid(action: Mapping, names: Iterable[str], errors: list[str], context: str) -> None:
    for name in names:
        if action.get(name) is not None:
            errors.append(f"{context} requires {name} to be null")


def _check_item_fields(action: Mapping, errors: list[str], context: str) -> None:
    """Attribute rules shared by NEW and UPDATE."""
    memory_type = action.get("memory_type")
    _require(action, ATTRIBUTES, errors, context)
    if memory_type == "intent":
        _require(action, ["stability"], errors, f"{context} intent")
        _forbid(action, ["anchor_date", "granularity"], errors, f"{context} intent")
        if (action.get("due_date") is None) != (action.get("due_granularity") is None):
            errors.append("due_date and due_granularity must both be set or both be null")
    else:
        _forbid(action, ["stability", "due_date", "due_granularity"], errors, f"{context} {memory_type}")
        if memory_type == "episodic":
            _require(action, ["anchor_date", "granularity"], errors, f"{context} episodic")
        else:
            _forbid(action, ["anchor_date", "granularity"], errors, f"{context} profile")


def _check_action_rules(action: Mapping, errors: list[str]) -> None:
    kind = action.get("action")
    memory_type = action.get("memory_type")
    context = str(kind)

    if kind == "NEW":
        _forbid(action, ["target_id", "reason", "intent_status"], errors, context)
        _check_item_fields(action, errors, context)
        return

    _require(action, ["target_id"], errors, context)

    if kind in ("RESOLVED", "ABANDONED") and memory_type != "intent":
        errors.append(f"{kind} applies only to intent items")

    if kind == "UPDATE":
        if action.get("reason") not in UPDATE_REASONS:
            errors.append("UPDATE requires reason change or correction")
        if memory_type != "intent":
            _forbid(action, ["intent_status"], errors, f"UPDATE {memory_type}")
        _check_item_fields(action, errors, context)
    elif kind == "REMOVE":
        allowed = INTENT_REMOVE_REASONS if memory_type == "intent" else REMOVE_REASONS
        if action.get("reason") not in allowed:
            errors.append(f"REMOVE of {memory_type} requires reason in {sorted(allowed)}")
        _forbid(action, [n for n in OPTIONAL_FIELDS if n != "reason"], errors, context)
    elif kind in ("CONFIRM", "ABANDONED"):
        _forbid(action, OPTIONAL_FIELDS, errors, context)
    elif kind == "RESOLVED":
        _forbid(action, ["reason", "intent_status", "stability", "due_date", "due_granularity"],
                errors, context)
        present = [action.get(n) is not None for n in EVENT_FIELDS]
        if any(present) and not all(present):
            errors.append("RESOLVED must fill all event fields "
                          f"({', '.join(EVENT_FIELDS)}) or leave all null")


def _check_dates(action: Mapping, session_date: date, errors: list[str]) -> None:
    if action.get("anchor_date") is not None:
        anchor = _parse_date(action["anchor_date"], action.get("granularity"), "anchor_date", errors)
        if anchor is not None and anchor > session_date:
            errors.append("anchor_date must not be after session_date")
    if action.get("due_date") is not None:
        due = _parse_date(action["due_date"], action.get("due_granularity"), "due_date", errors)
        granularity = action.get("due_granularity") or "day"
        if due is not None and period_end(due, granularity) < session_date:
            errors.append("due_date must not be before session_date")


def _check_target(action: Mapping, items_by_id: Mapping[str, tuple[str, Mapping]],
                  errors: list[str]) -> None:
    target_id = action.get("target_id")
    if target_id is None or not isinstance(target_id, str):
        return
    found = items_by_id.get(target_id)
    if found is None:
        errors.append(f"target_id {target_id!r} is not an existing item")
        return
    memory_type, item = found
    if action.get("memory_type") != memory_type:
        errors.append(f"memory_type {action.get('memory_type')!r} does not match target type {memory_type!r}")
    if (action.get("action") == "UPDATE" and item.get("privacy_level") == "HIGH"
            and action.get("privacy_level") == "LOW"):
        errors.append("UPDATE must not lower privacy_level from HIGH to LOW")


def validate_output(
    output: str | Mapping,
    *,
    session_date: str | date,
    conversation: str,
    profile_items: Iterable[Mapping] = (),
    episodic_items: Iterable[Mapping] = (),
    intent_items: Iterable[Mapping] = (),
) -> ValidationResult:
    """Split extractor output into actions APPLY may run and rejects.

    Raises OutputFormatError when the output as a whole is unusable.
    """
    if isinstance(output, str):
        output = parse_output(output)
    else:
        output = parse_output(json.dumps(output))
    if isinstance(session_date, str):
        session_date = date.fromisoformat(session_date)

    items_by_id: dict[str, tuple[str, Mapping]] = {}
    for memory_type, items in (("profile", profile_items), ("episodic", episodic_items),
                               ("intent", intent_items)):
        for item in items or ():
            items_by_id[item["id"]] = (memory_type, item)

    turns = [_normalize(turn) for turn in user_turns(conversation)]
    errors_by_index: list[list[str]] = []
    for action in output["actions"]:
        errors: list[str] = []
        if not isinstance(action, dict):
            errors_by_index.append(["action must be an object"])
            continue
        _check_shape(action, errors)
        if not errors:
            _check_action_rules(action, errors)
            _check_dates(action, session_date, errors)
            _check_target(action, items_by_id, errors)
            evidence = _normalize(action["evidence"])
            if not any(evidence in turn for turn in turns):
                errors.append("evidence is not a quote from a User turn")
        errors_by_index.append(errors)

    # Two actions on the same item leave APPLY with no well-defined order.
    targets: dict[str, list[int]] = {}
    for index, action in enumerate(output["actions"]):
        if isinstance(action, dict) and isinstance(action.get("target_id"), str):
            targets.setdefault(action["target_id"], []).append(index)
    for target_id, indexes in targets.items():
        if len(indexes) > 1:
            for index in indexes:
                errors_by_index[index].append(f"more than one action targets {target_id!r}")

    result = ValidationResult()
    for index, (action, errors) in enumerate(zip(output["actions"], errors_by_index)):
        if errors:
            result.rejects.append(Reject(index, action, errors))
        else:
            result.valid.append(action)
    return result
