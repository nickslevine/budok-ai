from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from tests.daemon._decision_fixtures import build_action, build_request
from yomi_daemon.prompt import (
    PromptTemplateVariant,
    available_prompt_versions,
    decision_output_json_schema,
    render_prompt,
)


def test_render_prompt_is_deterministic_for_same_request() -> None:
    request = build_request(
        (
            build_action("guard", description="Block safely."),
            build_action(
                "slash",
                payload_spec={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["target"],
                    "properties": {
                        "target": {
                            "type": "string",
                            "enum": ["enemy", "self"],
                            "semantic_hint": "throw_target",
                        }
                    },
                },
                prediction=True,
                prediction_spec={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["horizon"],
                    "properties": {
                        "horizon": {"type": "integer", "minimum": 1, "maximum": 3},
                        "confidence": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                },
                damage=80.0,
                startup_frames=5,
            ),
        )
    )

    first = render_prompt(
        request,
        configured_prompt_version="strategic_v1",
        policy_id="provider/openai-main",
    )
    second = render_prompt(
        request,
        configured_prompt_version="strategic_v1",
        policy_id="provider/openai-main",
    )

    assert first == second
    assert first.prompt_version == "strategic_v1"
    assert first.variant is PromptTemplateVariant.STRATEGIC
    assert "## Observation" in first.prompt_text
    assert '"policy_id": "provider/openai-main"' in first.prompt_text
    assert "`di: {x,y}` integers -100..100" in first.prompt_text
    assert '"prediction_spec"' in first.prompt_text


def test_render_prompt_prefers_request_prompt_version_over_policy_default() -> None:
    request = replace(
        build_request((build_action("guard"),)),
        prompt_version="few_shot_v1",
    )

    rendered = render_prompt(
        request,
        configured_prompt_version="minimal_v1",
        policy_id="provider/openai-main",
    )

    assert rendered.prompt_version == "few_shot_v1"
    assert rendered.variant is PromptTemplateVariant.FEW_SHOT
    assert "Follow these examples" in rendered.prompt_text


def test_action_only_rl_prompt_omits_reasoning_and_policy_metadata() -> None:
    request = build_request((build_action("guard"),))

    rendered = render_prompt(
        request,
        configured_prompt_version="strategic_rl_v1",
        policy_id="provider/openai-main",
    )

    assert rendered.prompt_version == "strategic_rl_v1"
    assert rendered.variant is PromptTemplateVariant.STRATEGIC
    assert '"policy_id": "provider/openai-main"' not in rendered.prompt_text
    assert '"trace_seed": 7' not in rendered.prompt_text
    assert "`reasoning`" not in rendered.prompt_text
    assert "`notes`" not in rendered.prompt_text
    assert 'Example: `{"action": "HSlash2"}`' in rendered.prompt_text


def test_available_prompt_versions_lists_current_template_inventory() -> None:
    assert available_prompt_versions() == (
        "few_shot_v1",
        "minimal_v1",
        "reasoning_v1",
        "strategic_rl_v1",
        "strategic_v1",
    )


def test_output_schema_di_range_matches_live_protocol() -> None:
    """DI range in prompt schema must match the [-100, 100] live protocol range."""
    schema = cast(Any, decision_output_json_schema())
    extra_props = schema["properties"]["extra"]["properties"]
    di_props = extra_props["di"]["properties"]
    assert di_props["x"]["minimum"] == -100
    assert di_props["x"]["maximum"] == 100
    assert di_props["y"]["minimum"] == -100
    assert di_props["y"]["maximum"] == 100


def test_output_schema_includes_prediction_support() -> None:
    """The output schema must describe prediction as an allowed extra."""
    schema = cast(Any, decision_output_json_schema())
    extra_props = schema["properties"]["extra"]["properties"]
    assert "prediction" in extra_props
    prediction_schema = extra_props["prediction"]
    # prediction is anyOf [prediction_spec, null]
    assert "anyOf" in prediction_schema
    type_options = [opt.get("type") for opt in prediction_schema["anyOf"]]
    assert "null" in type_options


def test_rendered_prompt_includes_structured_payload_spec() -> None:
    """Structured payload specs must appear verbatim in the rendered Legal Actions section."""
    payload_spec = cast(
        Any,
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["angle"],
            "properties": {
                "angle": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 360,
                    "semantic_hint": "aim_direction",
                }
            },
        },
    )
    request = build_request(
        (build_action("aim_shot", payload_spec=payload_spec, damage=60.0),)
    )

    rendered = render_prompt(request, configured_prompt_version="minimal_v1")

    assert '"semantic_hint": "aim_direction"' in rendered.prompt_text
    assert '"minimum": 0' in rendered.prompt_text
    assert '"maximum": 360' in rendered.prompt_text


def test_rendered_prompt_includes_prediction_spec_for_capable_action() -> None:
    """Actions with prediction support must have prediction_spec visible in the prompt."""
    prediction_spec = cast(
        Any,
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["horizon"],
            "properties": {
                "horizon": {"type": "integer", "minimum": 1, "maximum": 3},
                "opponent_action": {"type": "string"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
        },
    )
    request = build_request(
        (
            build_action(
                "read_attack",
                prediction=True,
                prediction_spec=prediction_spec,
            ),
        )
    )

    rendered = render_prompt(request, configured_prompt_version="minimal_v1")

    assert '"prediction_spec"' in rendered.prompt_text
    assert '"opponent_action"' in rendered.prompt_text
    assert '"confidence"' in rendered.prompt_text


def test_rendered_prompt_di_range_appears_in_output_contract() -> None:
    """The Output Contract section must show the full [-100, 100] DI range."""
    request = build_request((build_action("guard", di=True),))
    rendered = render_prompt(request, configured_prompt_version="minimal_v1")

    assert "`di: {x,y}` integers -100..100" in rendered.prompt_text


def test_rendered_prompt_includes_supports_flags() -> None:
    """Legal actions with supports flags must render them in the prompt."""
    request = build_request(
        (
            build_action(
                "feint_slash", di=True, feint=True, reverse=True, prediction=True
            ),
        )
    )

    rendered = render_prompt(request, configured_prompt_version="minimal_v1")
    prompt_text = rendered.prompt_text

    # The supports block should show all four flags
    assert '"di": true' in prompt_text
    assert '"feint": true' in prompt_text
    assert '"reverse": true' in prompt_text
    assert '"prediction": true' in prompt_text
