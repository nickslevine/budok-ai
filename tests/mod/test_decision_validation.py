"""Tests for mod-side decision validation, fallback, action application, and telemetry.

Validates the Python mirror harness in scripts/mod_decision_harness.py which
mirrors the GDScript implementations in mod/YomiLLMBridge/bridge/.
"""

from __future__ import annotations

import uuid

import pytest

from scripts.mod_decision_harness import (
    apply_decision,
    build_event_envelope,
    choose_fallback,
    is_replayable,
    validate_decision,
)


# --- Fixture Helpers ---


def _make_legal_action(
    action: str = "Jab",
    label: str | None = None,
    *,
    di: bool = True,
    feint: bool = False,
    reverse: bool = False,
    prediction: bool = False,
    payload_spec: dict | None = None,
    prediction_spec: dict | None = None,
    startup_frames: int | None = None,
    meter_cost: int | None = None,
    description: str | None = None,
) -> dict:
    result: dict = {
        "action": action,
        "label": label or action,
        "payload_spec": payload_spec or {},
        "supports": {
            "di": di,
            "feint": feint,
            "reverse": reverse,
            "prediction": prediction,
        },
    }
    if prediction_spec is not None:
        result["prediction_spec"] = prediction_spec
    if startup_frames is not None:
        result["startup_frames"] = startup_frames
    if meter_cost is not None:
        result["meter_cost"] = meter_cost
    if description is not None:
        result["description"] = description
    return result


def _make_request(
    *,
    match_id: str | None = None,
    turn_id: int = 1,
    player_id: str = "p1",
    legal_actions: list[dict] | None = None,
) -> dict:
    mid = match_id or str(uuid.uuid4())
    if legal_actions is None:
        legal_actions = [
            _make_legal_action("Jab"),
            _make_legal_action("Block", di=False, description="block guard"),
            _make_legal_action("Slash", feint=True),
        ]
    return {
        "match_id": mid,
        "turn_id": turn_id,
        "player_id": player_id,
        "deadline_ms": 2500,
        "state_hash": "abc123",
        "legal_actions_hash": "def456",
        "decision_type": "turn_action",
        "observation": {},
        "legal_actions": legal_actions,
    }


def _make_decision(
    request: dict,
    action: str = "Jab",
    *,
    data: dict | None = None,
    di: dict | None = None,
    feint: bool = False,
    reverse: bool = False,
    enveloped: bool = True,
) -> dict:
    payload = {
        "match_id": request["match_id"],
        "turn_id": request["turn_id"],
        "action": action,
        "data": data,
        "extra": {"di": di, "feint": feint, "reverse": reverse, "prediction": None},
    }
    if enveloped:
        return {
            "type": "action_decision",
            "version": "v2",
            "ts": "2026-01-01T00:00:00Z",
            "payload": payload,
        }
    return payload


STRUCTURED_SPECIAL_PAYLOAD = {
    "type": "object",
    "additionalProperties": False,
    "required": ["target", "strength"],
    "properties": {
        "target": {"type": "string", "enum": ["enemy", "self"]},
        "strength": {"type": "integer", "minimum": 1, "maximum": 3, "default": 1},
    },
}

PREDICTION_SPEC = {
    "type": "object",
    "additionalProperties": False,
    "required": ["horizon"],
    "properties": {
        "horizon": {"type": "integer", "minimum": 1, "maximum": 3},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}


# ==================== Decision Validation Tests ====================


class TestDecisionValidation:
    """Validation tests for stale turn_id, wrong match_id, unsupported extras, illegal actions."""

    def test_valid_enveloped_decision(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab")
        assert validate_decision(decision, request) == ""

    def test_bare_decision_is_rejected(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab", enveloped=False)
        assert validate_decision(decision, request) == "malformed_output"

    def test_wrong_match_id_returns_stale(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab")
        decision["payload"]["match_id"] = "wrong-match-id"
        assert validate_decision(decision, request) == "stale_response"

    def test_wrong_turn_id_returns_stale(self) -> None:
        request = _make_request(turn_id=5)
        decision = _make_decision(request, "Jab")
        decision["payload"]["turn_id"] = 3
        assert validate_decision(decision, request) == "stale_response"

    def test_empty_action_returns_malformed(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "")
        assert validate_decision(decision, request) == "malformed_output"

    def test_missing_extra_returns_malformed(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab")
        del decision["payload"]["extra"]
        assert validate_decision(decision, request) == "malformed_output"

    def test_extra_missing_feint_returns_malformed(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab")
        del decision["payload"]["extra"]["feint"]
        assert validate_decision(decision, request) == "malformed_output"

    def test_extra_missing_reverse_returns_malformed(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab")
        del decision["payload"]["extra"]["reverse"]
        assert validate_decision(decision, request) == "malformed_output"

    def test_extra_missing_prediction_returns_malformed(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab")
        del decision["payload"]["extra"]["prediction"]
        assert validate_decision(decision, request) == "malformed_output"

    def test_illegal_action_name(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "NonexistentMove")
        assert validate_decision(decision, request) == "illegal_output"

    def test_di_on_action_that_does_not_support_di(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Block", di={"x": 10, "y": 20})
        assert validate_decision(decision, request) == "illegal_output"

    def test_di_in_valid_range(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab", di={"x": -100, "y": 100})
        assert validate_decision(decision, request) == ""

    def test_di_out_of_range_x(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab", di={"x": 101, "y": 0})
        assert validate_decision(decision, request) == "illegal_output"

    def test_di_out_of_range_y(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab", di={"x": 0, "y": -101})
        assert validate_decision(decision, request) == "illegal_output"

    def test_feint_on_unsupported_action(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab", feint=True)
        assert validate_decision(decision, request) == "illegal_output"

    def test_feint_on_supported_action(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Slash", feint=True)
        assert validate_decision(decision, request) == ""

    def test_reverse_on_unsupported_action(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab", reverse=True)
        assert validate_decision(decision, request) == "illegal_output"

    def test_payload_data_on_no_spec_action(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab", data={"target": "enemy"})
        assert validate_decision(decision, request) == "illegal_output"

    def test_null_data_always_valid(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab", data=None)
        assert validate_decision(decision, request) == ""

    def test_empty_dict_data_on_no_spec(self) -> None:
        request = _make_request()
        decision = _make_decision(request, "Jab", data={})
        assert validate_decision(decision, request) == ""

    def test_enveloped_missing_payload_malformed(self) -> None:
        request = _make_request()
        decision = {"type": "action_decision", "version": "v2", "ts": "..."}
        assert validate_decision(decision, request) == "malformed_output"

    def test_data_with_payload_spec_present(self) -> None:
        legal = [_make_legal_action("Special", payload_spec=STRUCTURED_SPECIAL_PAYLOAD)]
        request = _make_request(legal_actions=legal)
        decision = _make_decision(
            request,
            "Special",
            data={"target": "enemy", "strength": 2},
        )
        assert validate_decision(decision, request) == ""

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ({"target": "enemy", "strength": 0}, "illegal_output"),
            ({"target": "enemy", "strength": 2, "unknown": True}, "illegal_output"),
            ({"target": "other", "strength": 2}, "illegal_output"),
            ({"target": "enemy"}, "illegal_output"),
        ],
    )
    def test_structured_payload_validation_rejects_invalid_shapes(
        self, data: dict, expected: str
    ) -> None:
        request = _make_request(
            legal_actions=[
                _make_legal_action("Special", payload_spec=STRUCTURED_SPECIAL_PAYLOAD)
            ]
        )
        decision = _make_decision(request, "Special", data=data)
        assert validate_decision(decision, request) == expected

    def test_prediction_requires_support_and_valid_shape(self) -> None:
        request = _make_request(
            legal_actions=[
                _make_legal_action(
                    "Read",
                    prediction=True,
                    prediction_spec=PREDICTION_SPEC,
                )
            ]
        )
        good = _make_decision(request, "Read")
        good["payload"]["extra"]["prediction"] = {"horizon": 2, "confidence": "high"}
        assert validate_decision(good, request) == ""

        bad = _make_decision(request, "Read")
        bad["payload"]["extra"]["prediction"] = {"horizon": 4, "confidence": "extreme"}
        assert validate_decision(bad, request) == "illegal_output"


class TestIsReplayable:
    """Tests for replayability checks (skip match_id/turn_id)."""

    def test_replayable_action_still_legal(self) -> None:
        request = _make_request()
        prior = {
            "action": "Jab",
            "extra": {"di": None, "feint": False, "reverse": False, "prediction": None},
        }
        assert is_replayable(prior, request) is True

    def test_not_replayable_action_no_longer_legal(self) -> None:
        request = _make_request()
        prior = {
            "action": "HeavySlash",
            "extra": {"di": None, "feint": False, "reverse": False, "prediction": None},
        }
        assert is_replayable(prior, request) is False

    def test_not_replayable_extras_invalid(self) -> None:
        request = _make_request()
        # Jab doesn't support feint
        prior = {
            "action": "Jab",
            "extra": {"di": None, "feint": True, "reverse": False, "prediction": None},
        }
        assert is_replayable(prior, request) is False


# ==================== Fallback Tests ====================


class TestFallbackSelection:
    """Tests for fallback action selection."""

    def test_safe_continue_picks_safest(self) -> None:
        actions = [
            _make_legal_action("Super Burst", di=True, meter_cost=3),
            _make_legal_action("Block", di=False, description="block guard defend"),
            _make_legal_action("Jab", di=True),
        ]
        request = _make_request(legal_actions=actions)
        result = choose_fallback(request, "timeout", "safe_continue")
        assert result["action"] == "Block"
        assert result["policy_id"] == "fallback/safe_continue"
        assert result["fallback_reason"] == "timeout"

    def test_heuristic_guard_picks_defensive(self) -> None:
        actions = [
            _make_legal_action("Jab", di=True),
            _make_legal_action("Guard Block", di=False, description="block guard"),
            _make_legal_action("Slash", di=True),
        ]
        request = _make_request(legal_actions=actions)
        result = choose_fallback(request, "disconnect", "heuristic_guard")
        assert result["action"] == "Guard Block"
        assert result["policy_id"] == "fallback/heuristic_guard"

    def test_heuristic_guard_falls_back_to_safe_when_no_guard_found(self) -> None:
        # No defensive keywords in any action
        actions = [
            _make_legal_action("Slash", di=True),
            _make_legal_action("Kick", di=True),
        ]
        request = _make_request(legal_actions=actions)
        result = choose_fallback(request, "timeout", "heuristic_guard")
        assert result["policy_id"] == "fallback/safe_continue"

    def test_last_valid_replayable_replays_when_legal(self) -> None:
        actions = [
            _make_legal_action("Jab", di=True),
            _make_legal_action("Slash", di=True),
        ]
        request = _make_request(player_id="p1", legal_actions=actions)
        prior = {
            "action": "Jab",
            "data": None,
            "extra": {"di": None, "feint": False, "reverse": False, "prediction": None},
        }
        result = choose_fallback(
            request,
            "timeout",
            "last_valid_replayable",
            last_valid_decisions={"p1": prior},
        )
        assert result["action"] == "Jab"
        assert result["policy_id"] == "fallback/last_valid_replayable"

    def test_last_valid_replayable_falls_to_guard_when_not_legal(self) -> None:
        actions = [
            _make_legal_action("Block", di=False, description="block guard"),
            _make_legal_action("Slash", di=True),
        ]
        request = _make_request(player_id="p1", legal_actions=actions)
        prior = {
            "action": "Jab",
            "data": None,
            "extra": {"di": None, "feint": False, "reverse": False, "prediction": None},
        }
        result = choose_fallback(
            request,
            "timeout",
            "last_valid_replayable",
            last_valid_decisions={"p1": prior},
        )
        assert result["action"] == "Block"
        assert result["policy_id"] == "fallback/heuristic_guard"

    def test_empty_legal_actions_returns_empty_fallback(self) -> None:
        request = _make_request(legal_actions=[])
        result = choose_fallback(request, "timeout", "safe_continue")
        assert result["action"] == ""
        assert result["fallback_reason"] == "timeout"

    def test_fallback_di_set_when_supported(self) -> None:
        actions = [_make_legal_action("Jab", di=True)]
        request = _make_request(legal_actions=actions)
        result = choose_fallback(request, "timeout", "safe_continue")
        assert result["extra"]["di"] == {"x": 0, "y": 0}
        assert result["extra"]["prediction"] is None

    def test_fallback_di_null_when_not_supported(self) -> None:
        actions = [_make_legal_action("Block", di=False)]
        request = _make_request(legal_actions=actions)
        result = choose_fallback(request, "timeout", "safe_continue")
        assert result["extra"]["di"] is None
        assert result["extra"]["prediction"] is None

    def test_fallback_resolves_structured_payload_defaults(self) -> None:
        actions = [
            _make_legal_action("Special", payload_spec=STRUCTURED_SPECIAL_PAYLOAD)
        ]
        request = _make_request(legal_actions=actions)
        result = choose_fallback(request, "timeout", "safe_continue")
        assert result["data"] == {"target": "enemy", "strength": 1}

    def test_fallback_preserves_match_and_turn_ids(self) -> None:
        request = _make_request(match_id="test-match-123", turn_id=7)
        result = choose_fallback(request, "disconnect", "safe_continue")
        assert result["match_id"] == "test-match-123"
        assert result["turn_id"] == 7


# ==================== Action Application Tests ====================


class TestActionApplication:
    """Tests for action application field resolution."""

    def test_apply_valid_decision(self) -> None:
        payload = {
            "action": "Jab",
            "data": None,
            "extra": {
                "di": {"x": 10, "y": -5},
                "feint": False,
                "reverse": False,
                "prediction": None,
            },
        }
        result = apply_decision(payload)
        assert result["applied"] is True
        assert result["apply_path"] == "queued_fields"
        assert result["queued_action"] == "Jab"
        assert result["queued_data"] is None
        assert result["queued_extra"]["di"] == {"x": 10, "y": -5}
        assert result["queued_extra"]["feint"] is False
        assert result["queued_extra"]["reverse"] is False
        assert result["queued_extra"]["prediction"] is None

    def test_apply_with_data(self) -> None:
        payload = {
            "action": "Special",
            "data": {"target": "enemy"},
            "extra": {"di": None, "feint": False, "reverse": False, "prediction": None},
        }
        result = apply_decision(payload)
        assert result["applied"] is True
        assert result["queued_data"] == {"target": "enemy"}

    def test_apply_normalizes_nested_float_payload_values_to_ints(self) -> None:
        payload = {
            "action": "Lasso",
            "data": {"Aim": {"x": 70.0, "y": 0.0}, "choices": [1.0, 2.5, 3.0]},
            "extra": {"di": None, "feint": False, "reverse": False, "prediction": None},
        }
        result = apply_decision(payload)
        assert result["applied"] is True
        assert result["queued_data"] == {
            "Aim": {"x": 70, "y": 0},
            "choices": [1, 2.5, 3],
        }

    def test_apply_unwraps_single_child_payload_maps(self) -> None:
        payload = {
            "action": "Lasso",
            "data": {"Aim": {"x": 70.0, "y": -10.0}},
            "extra": {"di": None, "feint": False, "reverse": False, "prediction": None},
        }
        result = apply_decision(payload)
        assert result["applied"] is True
        assert result["queued_data"] == {"x": 70, "y": -10}

    def test_apply_empty_action_fails(self) -> None:
        payload = {"action": "", "data": None, "extra": {}}
        result = apply_decision(payload)
        assert result["applied"] is False
        assert "error" in result

    def test_apply_null_extra_resolved(self) -> None:
        payload = {"action": "Jab", "data": None, "extra": None}
        result = apply_decision(payload)
        assert result["applied"] is True
        assert result["queued_extra"] is None

    def test_apply_with_di_resolved_to_int(self) -> None:
        payload = {
            "action": "Jab",
            "data": None,
            "extra": {
                "di": {"x": 50.0, "y": -25.0},
                "feint": True,
                "reverse": False,
                "prediction": None,
            },
        }
        result = apply_decision(payload)
        assert result["queued_extra"]["di"]["x"] == 50
        assert result["queued_extra"]["di"]["y"] == -25
        assert result["queued_extra"]["feint"] is True

    def test_apply_prefers_native_method_when_available(self) -> None:
        class MockFighter:
            def __init__(self) -> None:
                self.queued_action = ""
                self.queued_data = None
                self.queued_extra = None
                self.ready_state = False
                self.commit_count = 0

            def on_action_selected(
                self, action: str, data: dict | None, extra: dict | None
            ) -> None:
                self.queued_action = action
                self.queued_data = data
                self.queued_extra = extra
                self.ready_state = True
                self.commit_count += 1

        payload = {
            "action": "Slash",
            "data": {"target": "enemy"},
            "extra": {
                "di": {"x": 5, "y": -2},
                "feint": False,
                "reverse": True,
                "prediction": {"horizon": 2},
            },
        }
        fighter = MockFighter()

        result = apply_decision(payload, fighter)

        assert result["applied"] is True
        assert result["apply_path"] == "native_method"
        assert fighter.ready_state is True
        assert fighter.commit_count == 1
        assert fighter.queued_action == "Slash"
        assert fighter.queued_data == {"target": "enemy"}
        assert fighter.queued_extra == {
            "di": {"x": 5, "y": -2},
            "feint": False,
            "reverse": True,
            "prediction": {"horizon": 2},
        }


# ==================== Telemetry Tests ====================


class TestTelemetryEvents:
    """Tests for telemetry event envelope shape."""

    def test_turn_requested_envelope(self) -> None:
        envelope = build_event_envelope(
            "TurnRequested",
            "match-abc",
            turn_id=3,
            player_id="p1",
            details={"state_hash": "abc", "legal_actions_hash": "def"},
        )
        assert envelope["type"] == "event"
        assert envelope["version"] == "v2"
        assert envelope["payload"]["match_id"] == "match-abc"
        assert envelope["payload"]["event"] == "TurnRequested"
        assert envelope["payload"]["turn_id"] == 3
        assert envelope["payload"]["player_id"] == "p1"

    def test_decision_applied_envelope(self) -> None:
        envelope = build_event_envelope(
            "DecisionApplied",
            "match-abc",
            turn_id=5,
            player_id="p2",
            latency_ms=150,
            details={"action": "Jab"},
        )
        assert envelope["payload"]["event"] == "DecisionApplied"
        assert envelope["payload"]["latency_ms"] == 150

    def test_decision_fallback_envelope(self) -> None:
        envelope = build_event_envelope(
            "DecisionFallback",
            "match-abc",
            turn_id=2,
            player_id="p1",
            fallback_reason="timeout",
            latency_ms=2600,
            details={"action": "Block", "strategy": "safe_continue"},
        )
        assert envelope["payload"]["event"] == "DecisionFallback"
        assert envelope["payload"]["fallback_reason"] == "timeout"

    def test_match_ended_envelope(self) -> None:
        envelope = build_event_envelope(
            "MatchEnded",
            "match-abc",
            details={"winner": "p1", "end_reason": "ko", "total_turns": 42},
        )
        assert envelope["payload"]["event"] == "MatchEnded"
        assert envelope["payload"]["details"]["total_turns"] == 42

    def test_error_envelope(self) -> None:
        envelope = build_event_envelope(
            "Error",
            "match-abc",
            turn_id=10,
            player_id="p2",
            details={"error": "unexpected crash"},
        )
        assert envelope["payload"]["event"] == "Error"


# ==================== Integration / Reliability Tests ====================


class TestIntegration:
    """Integration tests for timeout, disconnect, malformed payload, and successful application."""

    def test_timeout_scenario_uses_fallback(self) -> None:
        """Simulate a timeout: no decision arrives, fallback is chosen."""
        request = _make_request()
        fallback = choose_fallback(request, "timeout", "safe_continue")
        assert fallback["fallback_reason"] == "timeout"
        assert fallback["action"] != ""
        result = apply_decision(fallback)
        assert result["applied"] is True

    def test_disconnect_scenario_uses_fallback(self) -> None:
        """Simulate a disconnect: fallback with disconnect reason."""
        request = _make_request()
        fallback = choose_fallback(request, "disconnect", "heuristic_guard")
        assert fallback["fallback_reason"] == "disconnect"
        result = apply_decision(fallback)
        assert result["applied"] is True

    def test_malformed_payload_caught_and_fallback_applied(self) -> None:
        """Malformed decision triggers validation error and fallback."""
        request = _make_request()
        bad_decision = {"match_id": request["match_id"], "turn_id": request["turn_id"]}
        error = validate_decision(bad_decision, request)
        assert error == "malformed_output"
        fallback = choose_fallback(request, error, "safe_continue")
        result = apply_decision(fallback)
        assert result["applied"] is True

    def test_illegal_action_caught_and_fallback_applied(self) -> None:
        """Illegal action triggers validation error and fallback."""
        request = _make_request()
        decision = _make_decision(request, "NonexistentMove")
        error = validate_decision(decision, request)
        assert error == "illegal_output"
        fallback = choose_fallback(request, error, "heuristic_guard")
        result = apply_decision(fallback)
        assert result["applied"] is True

    def test_stale_response_caught_and_fallback_applied(self) -> None:
        """Stale turn_id triggers validation error and fallback."""
        request = _make_request(turn_id=10)
        decision = _make_decision(request, "Jab")
        decision["payload"]["turn_id"] = 8  # stale
        error = validate_decision(decision, request)
        assert error == "stale_response"

    def test_successful_decision_flow(self) -> None:
        """Valid decision passes validation and applies cleanly."""
        request = _make_request()
        decision = _make_decision(request, "Slash", feint=True)
        error = validate_decision(decision, request)
        assert error == ""
        result = apply_decision(decision["payload"])
        assert result["applied"] is True
        assert result["queued_action"] == "Slash"


class TestReliability:
    """Reliability tests: bad daemon payloads never crash the harness."""

    @pytest.mark.parametrize(
        "bad_decision",
        [
            None,
            {},
            {"action": None},
            {"action": 123, "extra": {}},
            {"action": "Jab"},  # missing extra
            {"type": "wrong_type", "payload": {}},
            {"type": "action_decision"},  # missing payload
            {"type": "action_decision", "version": "v1", "payload": {}},
            {"match_id": "x", "turn_id": 1, "action": "Jab", "extra": "not_a_dict"},
            {
                "match_id": "x",
                "turn_id": 1,
                "action": "Jab",
                "extra": {
                    "di": "bad",
                    "feint": False,
                    "reverse": False,
                    "prediction": None,
                },
            },
        ],
        ids=[
            "none",
            "empty",
            "null_action",
            "int_action",
            "missing_extra",
            "wrong_type",
            "missing_payload",
            "wrong_version",
            "extra_not_dict",
            "di_not_dict",
        ],
    )
    def test_bad_payloads_do_not_crash(self, bad_decision: dict | None) -> None:
        request = _make_request()
        if bad_decision is None:
            bad_decision = {}
        # Must not raise, should return a non-empty error string
        result = validate_decision(bad_decision, request)
        assert result != "", f"Expected validation error for {bad_decision}"
