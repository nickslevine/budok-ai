"""Tests for Tinker environment wrapper."""

from __future__ import annotations

import json

import pytest

from yomi_daemon.protocol import ActionDecision, DIVector, DecisionExtras
from yomi_daemon.rl.tinker_env import (
    build_training_example,
    format_completion,
    format_completion_from_decision,
    format_prompt,
    parse_completion,
)

from tests.rl_fixtures import make_decision, make_request


class TestFormatPrompt:
    def test_uses_real_prompt_pipeline(self) -> None:
        """format_prompt delegates to prompt.render_prompt, so the output
        should contain sections from the real template (output contract,
        observation JSON, legal actions)."""
        request = make_request(player_id="p1", my_hp=80, opp_hp=60)
        prompt = format_prompt(request)

        # Output contract section from render_prompt
        assert "Output Contract" in prompt
        # Observation section
        assert "Observation" in prompt
        # The action "Wait" from our fixture should appear somewhere
        assert "Wait" in prompt

    def test_contains_action_instruction(self) -> None:
        request = make_request()
        prompt = format_prompt(request)

        assert '"action"' in prompt


class TestFormatCompletionFromDecision:
    def test_includes_action(self) -> None:
        decision = make_decision(action="HSlash2")
        result = format_completion_from_decision(decision)
        parsed = json.loads(result)
        assert parsed["action"] == "HSlash2"

    def test_includes_di_when_present(self) -> None:
        decision = ActionDecision(
            match_id="m1",
            turn_id=1,
            action="Wait",
            data=None,
            extra=DecisionExtras(
                di=DIVector(x=50, y=-30),
                feint=False,
                reverse=False,
            ),
        )
        result = format_completion_from_decision(decision)
        parsed = json.loads(result)
        assert parsed["extra"]["di"] == {"x": 50, "y": -30}

    def test_includes_feint_when_true(self) -> None:
        decision = ActionDecision(
            match_id="m1",
            turn_id=1,
            action="HSlash2",
            data=None,
            extra=DecisionExtras(di=None, feint=True, reverse=False),
        )
        result = format_completion_from_decision(decision)
        parsed = json.loads(result)
        assert parsed["extra"]["feint"] is True

    def test_omits_extra_when_all_defaults(self) -> None:
        decision = make_decision(action="Wait")
        result = format_completion_from_decision(decision)
        parsed = json.loads(result)
        assert "extra" not in parsed

    def test_includes_data_when_present(self) -> None:
        decision = ActionDecision(
            match_id="m1",
            turn_id=1,
            action="ParryHigh",
            data={"Melee Parry Timing": 10},
            extra=DecisionExtras(di=None, feint=False, reverse=False),
        )
        result = format_completion_from_decision(decision)
        parsed = json.loads(result)
        assert parsed["data"]["Melee Parry Timing"] == 10


class TestFormatCompletion:
    def test_simple_action(self) -> None:
        result = format_completion("HSlash2")
        parsed = json.loads(result)
        assert parsed["action"] == "HSlash2"

    def test_action_with_data(self) -> None:
        result = format_completion("Shoot", data={"direction": "forward"})
        parsed = json.loads(result)
        assert parsed["action"] == "Shoot"
        assert parsed["data"]["direction"] == "forward"


class TestParseCompletion:
    def test_parses_json_without_request(self) -> None:
        action, data = parse_completion('{"action": "HSlash2"}')
        assert action == "HSlash2"
        assert data is None

    def test_parses_json_with_data(self) -> None:
        action, data = parse_completion('{"action": "Shoot", "data": {"dir": "up"}}')
        assert action == "Shoot"
        assert data == {"dir": "up"}

    def test_parses_with_whitespace(self) -> None:
        action, data = parse_completion('  {"action": "Wait"}  ')
        assert action == "Wait"

    def test_raises_on_garbage(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            parse_completion("just some random text")

    def test_parses_json_in_multiline(self) -> None:
        text = 'I think the best move is:\n{"action": "ParryHigh"}\nBecause...'
        action, data = parse_completion(text)
        assert action == "ParryHigh"

    def test_validated_parse_with_request(self) -> None:
        """When request is provided, uses full daemon parser with normalization."""
        request = make_request()
        result = parse_completion('{"action": "Wait"}', request=request)
        # Should return a full ActionDecision
        assert isinstance(result, ActionDecision)
        assert result.action == "Wait"
        # Auto-filled fields from normalization
        assert result.match_id == "test-match"
        assert result.extra.feint is False

    def test_validated_parse_rejects_illegal_action(self) -> None:
        """When request is provided, illegal actions are rejected."""
        request = make_request()  # only has "Wait" as legal action
        with pytest.raises(ValueError):
            parse_completion('{"action": "NonexistentMove"}', request=request)


class TestBuildTrainingExample:
    def test_structure(self) -> None:
        request = make_request()
        decision = make_decision(action="Wait")
        example = build_training_example(request, decision, reward=0.5)

        assert "prompt" in example
        assert "completion" in example
        assert "reward" in example
        assert "metadata" in example

        assert example["reward"] == 0.5
        assert example["metadata"]["match_id"] == "test-match"

        # Completion should be valid JSON
        parsed = json.loads(example["completion"])
        assert parsed["action"] == "Wait"

    def test_prompt_matches_format_prompt(self) -> None:
        request = make_request()
        decision = make_decision(action="Wait")
        example = build_training_example(request, decision, reward=0.0)
        direct = format_prompt(request)

        assert example["prompt"] == direct

    def test_completion_includes_extras(self) -> None:
        request = make_request()
        decision = ActionDecision(
            match_id="test-match",
            turn_id=1,
            action="Wait",
            data=None,
            extra=DecisionExtras(
                di=DIVector(x=100, y=0),
                feint=False,
                reverse=False,
            ),
        )
        example = build_training_example(request, decision, reward=0.0)
        parsed = json.loads(example["completion"])
        assert parsed["extra"]["di"] == {"x": 100, "y": 0}
