"""Tinker environment wrapper for YOMI Hustle.

Maps our single-turn decision architecture to Tinker's ProblemEnv interface.
Each "problem" is one game state; the "answer" is the chosen action.
Reward comes from the step reward + (on final step) terminal reward.

This module defines the data contracts and completion formatting. Prompt
rendering reuses the existing `prompt.render_prompt` pipeline so the model
sees the same text during RL training as during live play.

Completion formatting includes the full output contract (action, data,
extra) so there is no train/inference mismatch. Parsing reuses the
daemon's `response_parser` for normalization and validation.

Actual Tinker client calls live in the training loop (not here) so this
module has zero runtime dependency on the `tinker` package.
"""

from __future__ import annotations

import json
from typing import Any

from yomi_daemon.prompt import render_prompt
from yomi_daemon.protocol import ActionDecision, DecisionRequest
from yomi_daemon.response_parser import (
    ResponseParsingError,
    parse_action_decision_response,
)


def format_prompt(
    request: DecisionRequest,
    *,
    prompt_version: str | None = None,
    policy_id: str | None = None,
) -> str:
    """Render the same prompt the model sees during live play.

    Delegates to `prompt.render_prompt` so training and inference prompts
    are always identical. Pass `prompt_version` to override (e.g.
    "minimal_v1", "strategic_v1").
    """
    rendered = render_prompt(
        request,
        configured_prompt_version=prompt_version,
        policy_id=policy_id,
    )
    return rendered.prompt_text


def format_completion_from_decision(decision: ActionDecision) -> str:
    """Format an ActionDecision as the completion string the model should produce.

    Includes all fields the live output contract expects: action, data, and
    extra (di, feint, reverse). This ensures the model is trained on the
    same output format it must produce at inference time.
    """
    obj: dict[str, Any] = {"action": decision.action}
    if decision.data is not None:
        obj["data"] = decision.data
    extra = decision.extra
    extra_obj: dict[str, Any] = {}
    if extra.di is not None:
        extra_obj["di"] = {"x": extra.di.x, "y": extra.di.y}
    if extra.feint:
        extra_obj["feint"] = True
    if extra.reverse:
        extra_obj["reverse"] = True
    if extra_obj:
        obj["extra"] = extra_obj
    return json.dumps(obj, separators=(",", ":"))


def format_completion(action: str, data: dict[str, Any] | None = None) -> str:
    """Format a simple action+data completion string.

    Prefer `format_completion_from_decision` when you have the full
    ActionDecision, as it includes extra fields (DI, feint, reverse).
    """
    obj: dict[str, Any] = {"action": action}
    if data:
        obj["data"] = data
    return json.dumps(obj, separators=(",", ":"))


def parse_completion(
    text: str,
    request: DecisionRequest | None = None,
) -> ActionDecision | tuple[str, dict[str, Any] | None]:
    """Parse a completion string into an ActionDecision or (action, data) tuple.

    When `request` is provided, uses the full daemon response parser with
    normalization (auto-fill match_id, turn_id, data defaults, extra defaults)
    and validation (legal action check, payload bounds). This matches what
    happens during live play.

    When `request` is None, falls back to simple JSON extraction returning
    (action_name, optional_data_dict). Raises ValueError on malformed input.
    """
    if request is not None:
        try:
            parsed = parse_action_decision_response(text, request)
            return parsed.decision
        except ResponseParsingError as exc:
            raise ValueError(str(exc)) from exc

    # Simple fallback: extract action + data from JSON
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action" in obj:
            return obj["action"], obj.get("data")
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "action" in obj:
                    return obj["action"], obj.get("data")
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse action from completion: {text!r}")


def build_training_example(
    request: DecisionRequest,
    decision: ActionDecision,
    reward: float,
    *,
    prompt_version: str | None = None,
    policy_id: str | None = None,
) -> dict[str, Any]:
    """Build a single training example for Tinker's GRPO loop.

    Returns a dict with:
        - prompt: the formatted game state text (same as live play)
        - completion: the chosen action as JSON (same format as live play)
        - reward: the scalar reward for this (state, action) pair
        - metadata: match/turn identifiers for debugging
    """
    return {
        "prompt": format_prompt(request, prompt_version=prompt_version, policy_id=policy_id),
        "completion": format_completion_from_decision(decision),
        "reward": reward,
        "metadata": {
            "match_id": request.match_id,
            "turn_id": request.turn_id,
            "player_id": request.player_id,
            "state_hash": request.state_hash,
        },
    }
