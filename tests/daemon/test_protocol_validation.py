from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from yomi_daemon.protocol import (
    ActionDecision,
    CharacterAssignments,
    CharacterSelectionConfig,
    CharacterSelectionMode,
    ConfigPayload,
    CURRENT_SCHEMA_VERSION,
    CURRENT_PROTOCOL_VERSION,
    DIVector,
    DecisionExtras,
    DecisionRequest,
    DecisionType,
    Envelope,
    Event,
    EventName,
    FallbackMode,
    FighterObservation,
    Hello,
    HelloAck,
    HistoryEntry,
    JsonObject,
    LegalAction,
    LegalActionSupports,
    LoggingConfig,
    MatchEnded,
    MessageType,
    Observation,
    PlayerPolicyMapping,
    ProtocolModel,
    ProtocolPayload,
    Vector2,
    canonical_json,
    canonical_sha256,
)
from yomi_daemon.validation import (
    DecisionValidationError,
    ProtocolValidationError,
    parse_envelope,
    validate_action_decision_for_request,
    validate_envelope,
    validate_model,
    validate_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_FIXTURES_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "protocol" / "canonical_hash_cases.json"
)


def build_config() -> ConfigPayload:
    return ConfigPayload(
        decision_timeout_ms=2500,
        fallback_mode=FallbackMode.SAFE_CONTINUE,
        logging=LoggingConfig(events=True, prompts=True, raw_provider_payloads=False),
        policy_mapping=PlayerPolicyMapping(
            p1="baseline/random", p2="baseline/block_always"
        ),
        character_selection=CharacterSelectionConfig(
            mode=CharacterSelectionMode.ASSIGNED,
            assignments=CharacterAssignments(p1="Cowboy", p2="Ninja"),
        ),
        stage_id="training_room",
    )


def build_hello() -> Hello:
    return Hello(
        game_version="1.0.0",
        mod_version="0.1.0",
        schema_version=CURRENT_SCHEMA_VERSION,
        supported_protocol_versions=(CURRENT_PROTOCOL_VERSION,),
    )


def build_hello_ack() -> HelloAck:
    return HelloAck(
        accepted_protocol_version=CURRENT_PROTOCOL_VERSION,
        accepted_schema_version=CURRENT_SCHEMA_VERSION,
        daemon_version="0.0.1",
        policy_mapping=PlayerPolicyMapping(
            p1="baseline/random", p2="baseline/block_always"
        ),
        config=build_config(),
    )


def build_observation() -> Observation:
    return Observation(
        tick=4242,
        frame=88,
        active_player="p1",
        fighters=(
            FighterObservation(
                id="p1",
                character="Cowboy",
                hp=960,
                max_hp=1000,
                meter=3,
                burst=1,
                position=Vector2(x=-12.5, y=0.0),
                velocity=Vector2(x=1.25, y=0.0),
                facing="right",
                current_state="neutral",
                combo_count=0,
                blockstun=0,
                hitlag=0,
                state_interruptable=True,
                can_feint=True,
                grounded=True,
            ),
            FighterObservation(
                id="p2",
                character="Ninja",
                hp=870,
                max_hp=1000,
                meter=2,
                burst=1,
                position=Vector2(x=12.5, y=0.0),
                velocity=Vector2(x=-0.5, y=0.0),
                facing="left",
                current_state="air_idle",
                combo_count=1,
                blockstun=0,
                hitlag=0,
                state_interruptable=True,
                can_feint=False,
                grounded=True,
            ),
        ),
        objects=(
            {
                "type": "Shuriken",
                "category": "projectile",
                "owner": "p2",
                "position": {"x": 5.0, "y": 3.0},
            },
        ),
        stage={"id": "training_room"},
        history=(HistoryEntry(turn_id=6, player_id="p2", action="jump"),),
    )


def build_legal_action() -> LegalAction:
    return LegalAction(
        action="slash",
        label="Slash",
        payload_spec={
            "type": "object",
            "additionalProperties": False,
            "required": ["target", "strength"],
            "properties": {
                "target": {
                    "type": "enemy",
                    "semantic_hint": "primary_target",
                },
                "strength": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "default": 1,
                    "ui_kind": "slider",
                },
            },
        },
        prediction_spec={
            "type": "object",
            "additionalProperties": False,
            "required": ["horizon"],
            "properties": {
                "horizon": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2,
                    "default": 1,
                },
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
        },
        payload_schema={"kind": "object", "fields": {"target": {"type": "string"}}},
        supports=LegalActionSupports(
            di=True, feint=False, reverse=True, prediction=True
        ),
        damage=120.0,
        startup_frames=5,
        range=18.5,
        meter_cost=1,
        description="Mid-range punish.",
    )


def build_decision_request() -> DecisionRequest:
    return DecisionRequest(
        match_id="match-001",
        turn_id=7,
        player_id="p1",
        deadline_ms=2500,
        state_hash="state-hash-001",
        legal_actions_hash="legal-hash-001",
        decision_type=DecisionType.TURN_ACTION,
        observation=build_observation(),
        legal_actions=(build_legal_action(),),
        trace_seed=17,
        game_version="1.0.0",
        mod_version="0.1.0",
        schema_version=CURRENT_SCHEMA_VERSION,
        ruleset_id="default-ruleset",
        prompt_version="strategic_v1",
    )


def build_action_decision() -> ActionDecision:
    return ActionDecision(
        match_id="match-001",
        turn_id=7,
        action="slash",
        data={"target": "enemy", "strength": 1},
        extra=DecisionExtras(
            di=DIVector(x=25, y=-10),
            feint=False,
            reverse=True,
            prediction={"horizon": 2, "confidence": "medium"},
        ),
        policy_id="baseline/random",
        latency_ms=123,
        tokens_in=0,
        tokens_out=0,
        notes="Deterministic baseline",
    )


def build_event() -> Event:
    return Event(
        match_id="match-001",
        event=EventName.DECISION_APPLIED,
        turn_id=7,
        player_id="p1",
        latency_ms=123,
        details={"action": "slash"},
    )


def build_match_ended() -> MatchEnded:
    return MatchEnded(
        match_id="match-001",
        winner="p1",
        end_reason="ko",
        total_turns=12,
        end_tick=5150,
        end_frame=129,
        replay_path="runs/20260312_match-001/replay.yomi",
        errors=(),
    )


def build_envelope(message_type: MessageType, payload: ProtocolPayload) -> Envelope:
    return Envelope(
        type=message_type,
        version=CURRENT_PROTOCOL_VERSION,
        ts="2026-03-12T00:00:00Z",
        payload=payload,
    )


VALID_CASES: list[tuple[MessageType, Callable[[], ProtocolModel]]] = [
    (MessageType.HELLO, build_hello),
    (MessageType.HELLO_ACK, build_hello_ack),
    (MessageType.DECISION_REQUEST, build_decision_request),
    (MessageType.ACTION_DECISION, build_action_decision),
    (MessageType.EVENT, build_event),
    (MessageType.MATCH_ENDED, build_match_ended),
    (MessageType.CONFIG, build_config),
]


@pytest.mark.parametrize(("message_type", "builder"), VALID_CASES)
def test_each_core_payload_validates(
    message_type: MessageType, builder: Callable[[], ProtocolModel]
) -> None:
    payload = builder().to_dict()
    validate_payload(message_type, payload)


@pytest.mark.parametrize(("message_type", "builder"), VALID_CASES)
def test_typed_models_round_trip_through_schema(
    message_type: MessageType,
    builder: Callable[[], ProtocolModel],
) -> None:
    model = builder()
    wire_payload = json.loads(json.dumps(model.to_dict()))

    validate_model(model)
    validate_payload(message_type, wire_payload)


def test_decision_request_schema_allows_negative_one_observation_frame() -> None:
    payload = build_decision_request().to_dict()
    observation_raw = payload["observation"]
    assert isinstance(observation_raw, dict)
    observation = dict(observation_raw)
    observation["frame"] = -1
    payload["observation"] = observation

    validate_payload(MessageType.DECISION_REQUEST, payload)


@pytest.mark.parametrize(
    ("message_type", "payload"),
    [
        (
            MessageType.HELLO,
            {
                "game_version": "1.0.0",
                "mod_version": "0.1.0",
                "schema_version": CURRENT_SCHEMA_VERSION,
            },
        ),
        (
            MessageType.HELLO_ACK,
            {
                "accepted_protocol_version": "v1",
                "accepted_schema_version": "v1",
                "daemon_version": "0.0.1",
                "policy_mapping": {
                    "p1": "baseline/random",
                    "p2": "baseline/block_always",
                },
            },
        ),
        (
            MessageType.DECISION_REQUEST,
            {
                **build_decision_request().to_dict(),
                "legal_actions": [],
            },
        ),
        (
            MessageType.ACTION_DECISION,
            {
                **build_action_decision().to_dict(),
                "extra": {
                    "di": {"x": 101, "y": 0},
                    "feint": False,
                    "reverse": False,
                    "prediction": None,
                },
            },
        ),
        (
            MessageType.EVENT,
            {"match_id": "match-001", "event": "TurnQueued"},
        ),
        (
            MessageType.MATCH_ENDED,
            {
                **build_match_ended().to_dict(),
                "total_turns": -1,
            },
        ),
        (
            MessageType.CONFIG,
            {
                **build_config().to_dict(),
                "fallback_mode": "panic",
            },
        ),
    ],
)
def test_each_core_payload_rejects_invalid_examples(
    message_type: MessageType,
    payload: dict[str, object],
) -> None:
    with pytest.raises(ProtocolValidationError):
        validate_payload(message_type, payload)


def test_envelope_validation_rejects_mismatched_version() -> None:
    envelope = build_envelope(
        MessageType.DECISION_REQUEST, build_decision_request()
    ).to_dict()
    envelope["version"] = "v1"

    with pytest.raises(ProtocolValidationError, match="version"):
        validate_envelope(envelope)


def test_validate_payload_rejects_explicit_unsupported_version() -> None:
    with pytest.raises(ProtocolValidationError, match="Unsupported protocol version"):
        validate_payload(
            MessageType.EVENT,
            build_event().to_dict(),
            version="v0",
        )


@pytest.mark.parametrize(
    "bad_di",
    [
        {"x": 101, "y": 0},
        {"x": 10.5, "y": 0},
        {"x": 0},
    ],
)
def test_action_decision_rejects_malformed_di_vectors(
    bad_di: dict[str, object],
) -> None:
    payload = build_action_decision().to_dict()
    payload["extra"] = {
        "di": bad_di,
        "feint": False,
        "reverse": True,
        "prediction": None,
    }

    with pytest.raises(ProtocolValidationError, match="di"):
        validate_payload(MessageType.ACTION_DECISION, payload)


@pytest.mark.parametrize(
    ("data", "match_text"),
    [
        ({"target": "enemy", "unknown": True}, "unknown"),
        ({"target": "enemy"}, "strength"),
        ({"target": "enemy", "strength": 0}, ">= 1"),
    ],
)
def test_request_relative_validation_rejects_invalid_payload_shapes(
    data: JsonObject,
    match_text: str,
) -> None:
    request = build_decision_request()
    decision = ActionDecision(
        match_id=request.match_id,
        turn_id=request.turn_id,
        action="slash",
        data=data,
        extra=DecisionExtras(
            di=None,
            feint=False,
            reverse=False,
            prediction={"horizon": 2, "confidence": "medium"},
        ),
    )

    with pytest.raises(DecisionValidationError, match=match_text):
        validate_action_decision_for_request(request, decision)


def test_request_relative_validation_rejects_invalid_prediction_payload() -> None:
    request = build_decision_request()
    decision = ActionDecision(
        match_id=request.match_id,
        turn_id=request.turn_id,
        action="slash",
        data={"target": "enemy", "strength": 1},
        extra=DecisionExtras(
            di=None,
            feint=False,
            reverse=False,
            prediction={"horizon": 3, "confidence": "medium"},
        ),
    )

    with pytest.raises(DecisionValidationError, match="prediction"):
        validate_action_decision_for_request(request, decision)


def test_parse_envelope_returns_typed_payload() -> None:
    envelope = build_envelope(
        MessageType.ACTION_DECISION, build_action_decision()
    ).to_dict()

    parsed = parse_envelope(envelope)

    assert parsed.type is MessageType.ACTION_DECISION
    assert parsed.version is CURRENT_PROTOCOL_VERSION
    assert isinstance(parsed.payload, ActionDecision)
    assert parsed.payload.extra.di == DIVector(x=25, y=-10)


def test_canonical_hash_fixtures_match_reference_values() -> None:
    fixtures = json.loads(CANONICAL_FIXTURES_PATH.read_text(encoding="utf-8"))

    for fixture in fixtures:
        assert canonical_json(fixture["payload"]) == fixture["canonical_json"]
        assert canonical_sha256(fixture["payload"]) == fixture["sha256"]
