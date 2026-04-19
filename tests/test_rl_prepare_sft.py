from __future__ import annotations

import json

from scripts.rl_prepare_sft import (
    format_completion_from_payload,
    render_prompt_from_request_payload,
)
from tests.daemon._decision_fixtures import build_action, build_request


def test_format_completion_from_payload_strips_reasoning_when_requested() -> None:
    request = build_request((build_action("guard"),))
    request_payload = request.to_dict()
    decision_payload = {
        "match_id": request.match_id,
        "turn_id": request.turn_id,
        "action": "guard",
        "data": None,
        "extra": {"di": None, "feint": False, "reverse": False, "prediction": None},
        "reasoning": "safe block while we are unsure",
    }

    stripped = format_completion_from_payload(
        request_payload,
        decision_payload,
        strip_reasoning=True,
    )
    full = format_completion_from_payload(
        request_payload,
        decision_payload,
        strip_reasoning=False,
    )

    assert json.loads(stripped) == {"action": "guard"}
    assert json.loads(full) == {
        "action": "guard",
        "reasoning": "safe block while we are unsure",
    }


def test_render_prompt_from_request_payload_supports_action_only_prompt_version() -> (
    None
):
    request = build_request((build_action("guard"),))

    prompt_text = render_prompt_from_request_payload(
        request.to_dict(),
        prompt_version="strategic_rl_v1",
    )

    assert '"policy_id"' not in prompt_text
    assert "`reasoning`" not in prompt_text
    assert 'Example: `{"action": "HSlash2"}`' in prompt_text
