# ruff: noqa: E402

"""Python mirror of the mod-side decision validation, fallback, and action application pipeline.

Operates on synthetic decision/request dicts to validate:
- Decision validation against pending requests and legal action sets
- Fallback action selection with guard/safe scoring heuristics
- Action application field resolution
- Telemetry event envelope shape

This harness mirrors the GDScript implementations in:
- mod/YomiLLMBridge/bridge/DecisionValidator.gd
- mod/YomiLLMBridge/bridge/FallbackHandler.gd
- mod/YomiLLMBridge/bridge/ActionApplier.gd
- mod/YomiLLMBridge/bridge/Telemetry.gd
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DAEMON_SRC = REPO_ROOT / "daemon" / "src"
if str(DAEMON_SRC) not in sys.path:
    sys.path.insert(0, str(DAEMON_SRC))

from yomi_daemon.protocol import CURRENT_PROTOCOL_VERSION, default_prediction_spec

DI_MIN = -100
DI_MAX = 100


# --- Decision Validation ---


def validate_decision(decision: dict[str, Any], request: dict[str, Any]) -> str:
    """Validate an action decision against a pending request.

    Returns "" on success or a fallback reason string on failure.
    """
    envelope_error = _validate_envelope_shape(decision)
    if envelope_error:
        return envelope_error

    payload = decision["payload"]

    # Match ID check
    if str(payload.get("match_id", "")) != str(request.get("match_id", "")):
        return "stale_response"

    # Turn ID check
    if int(payload.get("turn_id", -1)) != int(request.get("turn_id", -1)):
        return "stale_response"

    # Action must be a non-empty string
    action_name = str(payload.get("action", ""))
    if not action_name:
        return "malformed_output"

    # Extra must be a dict with feint and reverse
    extra = payload.get("extra")
    if extra is None or not isinstance(extra, dict):
        return "malformed_output"
    if "feint" not in extra or "reverse" not in extra or "prediction" not in extra:
        return "malformed_output"

    # Find matching legal action
    legal_actions = request.get("legal_actions", [])
    matched_action = _find_legal_action(action_name, legal_actions)
    if matched_action is None:
        return "illegal_output"

    # Validate extras against supports
    extras_error = _validate_extras(extra, matched_action)
    if extras_error:
        return extras_error

    # Validate payload data
    data_error = _validate_payload_data(payload.get("data"), matched_action)
    if data_error:
        return data_error

    return ""


def is_replayable(decision_payload: dict[str, Any], request: dict[str, Any]) -> bool:
    """Check if a prior decision can be replayed under the current legal set."""
    action_name = str(decision_payload.get("action", ""))
    if not action_name:
        return False

    extra = decision_payload.get("extra")
    if extra is None or not isinstance(extra, dict):
        return False

    legal_actions = request.get("legal_actions", [])
    matched_action = _find_legal_action(action_name, legal_actions)
    if matched_action is None:
        return False

    if _validate_extras(extra, matched_action):
        return False

    if _validate_payload_data(decision_payload.get("data"), matched_action):
        return False

    return True


def _validate_envelope_shape(decision: dict[str, Any]) -> str:
    if str(decision.get("type", "")) != "action_decision":
        return "malformed_output"
    if str(decision.get("version", "")) != CURRENT_PROTOCOL_VERSION.value:
        return "malformed_output"
    if "payload" not in decision or not isinstance(decision.get("payload"), dict):
        return "malformed_output"
    return ""


def _find_legal_action(
    action_name: str, legal_actions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for legal_action in legal_actions:
        if not isinstance(legal_action, dict):
            continue
        if str(legal_action.get("action", "")) == action_name:
            return legal_action
    return None


def _validate_extras(extra: dict[str, Any], legal_action: dict[str, Any]) -> str:
    supports = legal_action.get("supports", {})
    if not isinstance(supports, dict):
        return "malformed_output"

    di = extra.get("di")
    if di is not None:
        if not bool(supports.get("di", False)):
            return "illegal_output"
        if not isinstance(di, dict):
            return "malformed_output"
        di_x = di.get("x")
        di_y = di.get("y")
        if di_x is None or di_y is None:
            return "malformed_output"
        if int(di_x) < DI_MIN or int(di_x) > DI_MAX:
            return "illegal_output"
        if int(di_y) < DI_MIN or int(di_y) > DI_MAX:
            return "illegal_output"

    if bool(extra.get("feint", False)) and not bool(supports.get("feint", False)):
        return "illegal_output"

    if bool(extra.get("reverse", False)) and not bool(supports.get("reverse", False)):
        return "illegal_output"
    prediction = extra.get("prediction")
    if prediction is not None and not bool(supports.get("prediction", False)):
        return "illegal_output"
    if prediction is not None:
        if not isinstance(prediction, dict):
            return "malformed_output"
        prediction_spec = legal_action.get("prediction_spec")
        if not isinstance(prediction_spec, dict):
            prediction_spec = default_prediction_spec()
        error = _validate_payload_value(
            prediction,
            prediction_spec,
            context="extra.prediction",
            field_map_mode=False,
        )
        if error:
            return error

    return ""


def _validate_payload_data(data: Any, legal_action: dict[str, Any]) -> str:
    payload_spec = legal_action.get("payload_spec", {})
    if not isinstance(payload_spec, dict) or not payload_spec:
        if isinstance(data, dict) and data:
            return "illegal_output"
        return ""

    if data is None:
        return "illegal_output"

    if not isinstance(data, dict):
        return "illegal_output"

    return _validate_payload_value(
        data,
        payload_spec,
        context="data",
        field_map_mode=_looks_like_field_map(payload_spec),
    )


# --- Fallback Selection ---

DEFENSIVE_KEYWORDS = (
    "block",
    "guard",
    "parry",
    "defend",
    "shield",
    "wait",
    "idle",
    "evade",
    "backdash",
)
LOW_COMMITMENT_KEYWORDS = (
    "block",
    "guard",
    "wait",
    "backdash",
    "jab",
    "poke",
    "dash",
    "step",
    "move",
)
HIGH_COMMITMENT_KEYWORDS = ("super", "burst", "charge", "launcher", "throw", "teleport")


def choose_fallback(
    request: dict[str, Any],
    fallback_reason: str,
    fallback_mode: str,
    last_valid_decisions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Choose a fallback action for the given request."""
    legal_actions = request.get("legal_actions", [])
    if not legal_actions:
        return _empty_fallback(request, fallback_reason)

    player_id = str(request.get("player_id", ""))

    if fallback_mode == "last_valid_replayable" and last_valid_decisions:
        last_decision = last_valid_decisions.get(player_id)
        if last_decision is not None and is_replayable(last_decision, request):
            return _build_fallback_from_prior(request, last_decision, fallback_reason)

    if fallback_mode in ("heuristic_guard", "last_valid_replayable"):
        guard_action = _select_guard_action(legal_actions)
        if guard_action is not None:
            return _build_fallback_from_action(
                request, guard_action, fallback_reason, "heuristic_guard"
            )

    safe_action = _select_safe_action(legal_actions)
    return _build_fallback_from_action(
        request, safe_action, fallback_reason, "safe_continue"
    )


def _empty_fallback(request: dict[str, Any], fallback_reason: str) -> dict[str, Any]:
    return {
        "match_id": request.get("match_id", ""),
        "turn_id": request.get("turn_id", 0),
        "action": "",
        "data": None,
        "extra": {"di": None, "feint": False, "reverse": False, "prediction": None},
        "policy_id": "fallback/safe_continue",
        "notes": "No legal actions available for fallback.",
        "fallback_reason": fallback_reason,
    }


def _build_fallback_from_prior(
    request: dict[str, Any],
    prior_decision: dict[str, Any],
    fallback_reason: str,
) -> dict[str, Any]:
    return {
        "match_id": request.get("match_id", ""),
        "turn_id": request.get("turn_id", 0),
        "action": prior_decision.get("action", ""),
        "data": prior_decision.get("data"),
        "extra": prior_decision.get(
            "extra", {"di": None, "feint": False, "reverse": False, "prediction": None}
        ),
        "policy_id": "fallback/last_valid_replayable",
        "notes": "Replayed the last request-compatible decision after upstream failure.",
        "fallback_reason": fallback_reason,
    }


def _build_fallback_from_action(
    request: dict[str, Any],
    action: dict[str, Any],
    fallback_reason: str,
    strategy: str,
) -> dict[str, Any]:
    supports = action.get("supports", {})
    di_value = {"x": 0, "y": 0} if bool(supports.get("di", False)) else None

    return {
        "match_id": request.get("match_id", ""),
        "turn_id": request.get("turn_id", 0),
        "action": str(action.get("action", "")),
        "data": _resolve_payload_object(action.get("payload_spec", {})),
        "extra": {
            "di": di_value,
            "feint": False,
            "reverse": False,
            "prediction": None,
        },
        "policy_id": f"fallback/{strategy}",
        "notes": "Selected fallback action after upstream failure.",
        "fallback_reason": fallback_reason,
    }


def _action_text(action: dict[str, Any]) -> str:
    parts = [str(action.get("action", ""))]
    label = action.get("label")
    if label is not None and str(label):
        parts.append(str(label))
    description = action.get("description")
    if description is not None and str(description):
        parts.append(str(description))
    return " ".join(parts).lower()


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _guard_score(action: dict[str, Any]) -> float:
    text = _action_text(action)
    return float(_keyword_hits(text, DEFENSIVE_KEYWORDS) * 4) + _safe_score(action)


def _safe_score(action: dict[str, Any]) -> float:
    text = _action_text(action)
    payload_spec = action.get("payload_spec", {})
    supports = action.get("supports", {})

    no_payload_bonus = 3.0 if not payload_spec else 0.0
    no_extra_bonus = 1.0 if not bool(supports.get("di", False)) else 0.0
    defensive_bonus = float(_keyword_hits(text, DEFENSIVE_KEYWORDS) * 4)
    low_commitment_bonus = float(_keyword_hits(text, LOW_COMMITMENT_KEYWORDS) * 2)
    high_commitment_penalty = float(_keyword_hits(text, HIGH_COMMITMENT_KEYWORDS) * 3)

    startup_frames = action.get("startup_frames")
    startup_value = int(startup_frames) if startup_frames is not None else 999
    startup_penalty = min(startup_value, 30) / 10.0

    meter_cost = action.get("meter_cost")
    meter_penalty = float(int(meter_cost) * 2) if meter_cost is not None else 0.0

    return (
        no_payload_bonus
        + no_extra_bonus
        + defensive_bonus
        + low_commitment_bonus
        - high_commitment_penalty
        - startup_penalty
        - meter_penalty
    )


def _select_guard_action(legal_actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [(i, a) for i, a in enumerate(legal_actions) if _guard_score(a) > 0]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda pair: (_guard_score(pair[1]), _safe_score(pair[1]), -pair[0]),
    )[1]


def _select_safe_action(legal_actions: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        enumerate(legal_actions),
        key=lambda pair: (_safe_score(pair[1]), -pair[0]),
    )[1]


_SCHEMA_DESCRIPTOR_KEYS = frozenset(
    {
        "additionalProperties",
        "choices",
        "const",
        "default",
        "enum",
        "items",
        "maximum",
        "maxItems",
        "max_items",
        "maxLength",
        "minimum",
        "minItems",
        "min_items",
        "minLength",
        "properties",
        "required",
        "semantic_hint",
        "type",
        "ui_kind",
    }
)


def _looks_like_field_map(spec: dict[str, Any]) -> bool:
    return not bool(_SCHEMA_DESCRIPTOR_KEYS & set(spec))


def _validate_payload_value(
    value: Any,
    spec: dict[str, Any],
    *,
    context: str,
    field_map_mode: bool = False,
) -> str:
    if field_map_mode:
        if not isinstance(value, dict):
            return "illegal_output"
        for key in value:
            if key not in spec:
                return "illegal_output"
        for key, item in value.items():
            descriptor = spec.get(key)
            if isinstance(descriptor, dict):
                error = _validate_payload_value(
                    item,
                    descriptor,
                    context=f"{context}.{key}",
                    field_map_mode=_looks_like_field_map(descriptor),
                )
                if error:
                    return error
        for key, descriptor in spec.items():
            if (
                isinstance(descriptor, dict)
                and bool(descriptor.get("required", False))
                and key not in value
            ):
                return "illegal_output"
        return ""

    type_error = _validate_descriptor_type(value, spec)
    if type_error:
        return type_error

    if "const" in spec and value != spec["const"]:
        return "illegal_output"

    enum_values = spec.get("enum", spec.get("choices"))
    if isinstance(enum_values, list) and value not in enum_values:
        return "illegal_output"

    if isinstance(value, dict):
        properties = spec.get("properties")
        if isinstance(properties, dict):
            additional_properties = spec.get("additionalProperties", True)
            for key, item in value.items():
                if key in properties:
                    descriptor = properties[key]
                    if isinstance(descriptor, dict):
                        error = _validate_payload_value(
                            item,
                            descriptor,
                            context=f"{context}.{key}",
                            field_map_mode=_looks_like_field_map(descriptor),
                        )
                        if error:
                            return error
                    continue
                if additional_properties is False:
                    return "illegal_output"

            required = spec.get("required")
            if isinstance(required, list):
                for required_key in required:
                    if isinstance(required_key, str) and required_key not in value:
                        return "illegal_output"

    if isinstance(value, list):
        items_spec = spec.get("items")
        if isinstance(items_spec, dict):
            for item in value:
                error = _validate_payload_value(
                    item,
                    items_spec,
                    context=f"{context}[]",
                    field_map_mode=_looks_like_field_map(items_spec),
                )
                if error:
                    return error
        min_items = _optional_int(spec.get("minItems", spec.get("min_items")))
        if min_items is not None and len(value) < min_items:
            return "illegal_output"
        max_items = _optional_int(spec.get("maxItems", spec.get("max_items")))
        if max_items is not None and len(value) > max_items:
            return "illegal_output"

    if _is_number(value):
        minimum = _optional_number(spec.get("minimum"))
        if minimum is not None and float(value) < minimum:
            return "illegal_output"
        maximum = _optional_number(spec.get("maximum"))
        if maximum is not None and float(value) > maximum:
            return "illegal_output"

    if isinstance(value, str):
        min_length = _optional_int(spec.get("minLength"))
        if min_length is not None and len(value) < min_length:
            return "illegal_output"
        max_length = _optional_int(spec.get("maxLength"))
        if max_length is not None and len(value) > max_length:
            return "illegal_output"

    return ""


def _validate_descriptor_type(value: Any, spec: dict[str, Any]) -> str:
    normalized_type = str(spec.get("type", "")).lower()
    if normalized_type == "":
        return ""
    if normalized_type == "string":
        return "" if isinstance(value, str) else "illegal_output"
    if normalized_type == "integer":
        return (
            ""
            if isinstance(value, int) and not isinstance(value, bool)
            else "illegal_output"
        )
    if normalized_type == "number":
        return "" if _is_number(value) else "illegal_output"
    if normalized_type == "boolean":
        return "" if isinstance(value, bool) else "illegal_output"
    if normalized_type == "object":
        return "" if isinstance(value, dict) else "illegal_output"
    if normalized_type == "array":
        return "" if isinstance(value, list) else "illegal_output"
    return "" if isinstance(value, str) else "illegal_output"


def _resolve_payload_object(spec: Any) -> dict[str, Any] | None:
    if not isinstance(spec, dict) or not spec:
        return None
    resolved = _resolve_schema_value(spec)
    return resolved if isinstance(resolved, dict) else None


def _resolve_schema_value(spec: Any) -> Any:
    if spec is None or isinstance(spec, (bool, int, float, str)):
        return spec
    if not isinstance(spec, dict):
        return None

    if "const" in spec:
        return spec["const"]
    if "default" in spec:
        return spec["default"]

    enum_values = spec.get("enum", spec.get("choices"))
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    normalized_type = str(spec.get("type", "")).lower()
    if normalized_type == "object" or "properties" in spec:
        properties = spec.get("properties", {})
        if not isinstance(properties, dict):
            return {}
        return {key: _resolve_schema_value(value) for key, value in properties.items()}
    if normalized_type == "array":
        items_spec = spec.get("items")
        min_items = _optional_int(spec.get("minItems", spec.get("min_items", 0))) or 0
        if items_spec is None:
            return []
        return [_resolve_schema_value(items_spec) for _ in range(max(0, min_items))]
    if normalized_type == "integer":
        return _optional_int(spec.get("minimum", spec.get("min", 0))) or 0
    if normalized_type == "number":
        return _optional_number(spec.get("minimum", spec.get("min", 0.0))) or 0.0
    if normalized_type == "boolean":
        return False
    if normalized_type in {"enemy", "opponent", "target_enemy"}:
        return "enemy"
    if normalized_type in {"self", "ally", "target_self"}:
        return "self"
    if normalized_type in {"direction", "facing"}:
        choices = spec.get("enum", ["forward"])
        return choices[0] if isinstance(choices, list) and choices else "forward"
    if normalized_type in {"vector2", "xy"}:
        return {"x": 0, "y": 0}
    if normalized_type == "string":
        if "example" in spec:
            return str(spec["example"])
        if "placeholder" in spec:
            return str(spec["placeholder"])
        min_length = _optional_int(spec.get("minLength")) or 0
        return "x" * max(0, min_length)
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_number(value: Any) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


# --- Action Application ---


def apply_decision(
    decision_payload: dict[str, Any], fighter: Any | None = None
) -> dict[str, Any]:
    """Resolve the fields that would be written to a fighter node.

    When a fighter exposes the native `on_action_selected` hook, prefer that path
    so ready-state side effects match the in-engine contract.
    """
    action_name = str(decision_payload.get("action", ""))
    if not action_name:
        return {"applied": False, "error": "empty action name"}

    queued_data = _resolve_queued_data(decision_payload.get("data"))
    queued_extra = _resolve_queued_extra(decision_payload.get("extra"))

    if fighter is not None and hasattr(fighter, "on_action_selected"):
        fighter.on_action_selected(action_name, queued_data, queued_extra)
        return {
            "applied": True,
            "action": action_name,
            "apply_path": "native_method",
            "queued_action": getattr(fighter, "queued_action", action_name),
            "queued_data": getattr(fighter, "queued_data", queued_data),
            "queued_extra": getattr(fighter, "queued_extra", queued_extra),
        }

    return {
        "applied": True,
        "action": action_name,
        "apply_path": "queued_fields",
        "queued_action": action_name,
        "queued_data": queued_data,
        "queued_extra": queued_extra,
    }


def _resolve_queued_data(data: Any) -> Any:
    if data is None:
        return None
    if isinstance(data, dict):
        if len(data) == 1:
            only_value = next(iter(data.values()))
            if isinstance(only_value, dict):
                return _normalize_game_value(only_value)
        return _normalize_game_value(data)
    return None


def _normalize_game_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_game_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_game_value(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _resolve_queued_extra(extra: Any) -> dict[str, Any] | None:
    if extra is None or not isinstance(extra, dict):
        return None

    resolved: dict[str, Any] = {}
    di = extra.get("di")
    if di is not None and isinstance(di, dict):
        resolved["di"] = {"x": int(di.get("x", 0)), "y": int(di.get("y", 0))}
    else:
        resolved["di"] = None

    resolved["feint"] = bool(extra.get("feint", False))
    resolved["reverse"] = bool(extra.get("reverse", False))
    resolved["prediction"] = (
        dict(extra["prediction"]) if isinstance(extra.get("prediction"), dict) else None
    )
    return resolved


# --- Telemetry Event Building ---


def build_event_envelope(
    event_name: str,
    match_id: str,
    *,
    turn_id: int | None = None,
    player_id: str | None = None,
    fallback_reason: str | None = None,
    latency_ms: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a telemetry event envelope matching the protocol Event shape."""
    payload: dict[str, Any] = {
        "match_id": match_id,
        "event": event_name,
    }
    if turn_id is not None:
        payload["turn_id"] = turn_id
    if player_id is not None:
        payload["player_id"] = player_id
    if fallback_reason is not None:
        payload["fallback_reason"] = fallback_reason
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    if details is not None:
        payload["details"] = details

    return {
        "type": "event",
        "version": CURRENT_PROTOCOL_VERSION.value,
        "ts": "2026-01-01T00:00:00Z",  # placeholder for tests
        "payload": payload,
    }
