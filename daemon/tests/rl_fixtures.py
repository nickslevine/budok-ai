"""Test fixtures for RL module tests."""

from __future__ import annotations

from yomi_daemon.protocol import (
    ActionDecision,
    DecisionExtras,
    DecisionRequest,
    DecisionType,
    FighterObservation,
    LegalAction,
    LegalActionSupports,
    MatchEnded,
    Observation,
    Vector2,
)
from yomi_daemon.rl.trajectory import Step


def make_step(
    *,
    my_hp: int = 100,
    opp_hp: int = 100,
    max_hp: int = 100,
    my_position_x: float = 0.0,
    my_position_y: float = 0.0,
    opp_position_x: float = 200.0,
    opp_position_y: float = 0.0,
    my_combo_count: int = 0,
    opp_combo_count: int = 0,
    opp_blockstun: int = 0,
    my_meter: int = 0,
    opp_meter: int = 0,
    my_burst: int = 1,
    opp_burst: int = 1,
    stage_width: float = 1100.0,
    my_hp_delta: int | None = None,
    opp_hp_delta: int | None = None,
    my_outcome: str | None = None,
    opp_outcome: str | None = None,
    was_fallback: bool = False,
    action: str = "Wait",
    turn_id: int = 1,
    tick: int = 10,
    latency_ms: int | None = None,
) -> Step:
    """Create a Step with sensible defaults for testing."""
    request = make_request(
        player_id="p1",
        turn_id=turn_id,
        my_hp=my_hp,
        opp_hp=opp_hp,
        max_hp=max_hp,
        my_position_x=my_position_x,
        my_position_y=my_position_y,
        opp_position_x=opp_position_x,
        opp_position_y=opp_position_y,
        my_combo_count=my_combo_count,
        opp_combo_count=opp_combo_count,
        opp_blockstun=opp_blockstun,
        my_meter=my_meter,
        opp_meter=opp_meter,
        my_burst=my_burst,
        opp_burst=opp_burst,
        stage_width=stage_width,
        tick=tick,
    )
    decision = make_decision(
        action=action,
        turn_id=turn_id,
        was_fallback=was_fallback,
        latency_ms=latency_ms,
    )

    return Step(
        turn_id=turn_id,
        tick=tick,
        request=request,
        decision=decision,
        my_hp=my_hp,
        opp_hp=opp_hp,
        max_hp=max_hp,
        my_position_x=my_position_x,
        my_position_y=my_position_y,
        opp_position_x=opp_position_x,
        opp_position_y=opp_position_y,
        my_combo_count=my_combo_count,
        opp_combo_count=opp_combo_count,
        opp_blockstun=opp_blockstun,
        my_meter=my_meter,
        opp_meter=opp_meter,
        my_burst=my_burst,
        opp_burst=opp_burst,
        stage_width=stage_width,
        my_hp_delta=my_hp_delta,
        opp_hp_delta=opp_hp_delta,
        my_outcome=my_outcome,
        opp_outcome=opp_outcome,
        was_fallback=was_fallback,
        action=action,
        latency_ms=latency_ms,
    )


def make_request(
    *,
    player_id: str = "p1",
    turn_id: int = 1,
    my_hp: int = 100,
    opp_hp: int = 100,
    max_hp: int = 100,
    my_position_x: float = 0.0,
    my_position_y: float = 0.0,
    opp_position_x: float = 200.0,
    opp_position_y: float = 0.0,
    my_combo_count: int = 0,
    opp_combo_count: int = 0,
    opp_blockstun: int = 0,
    my_meter: int = 0,
    opp_meter: int = 0,
    my_burst: int = 1,
    opp_burst: int = 1,
    stage_width: float = 1100.0,
    tick: int = 10,
) -> DecisionRequest:
    """Create a DecisionRequest with sensible defaults."""
    opp_id = "p2" if player_id == "p1" else "p1"

    me = FighterObservation(
        id=player_id,
        character="Ninja",
        hp=my_hp,
        max_hp=max_hp,
        meter=my_meter,
        burst=my_burst,
        position=Vector2(x=my_position_x, y=my_position_y),
        velocity=Vector2(x=0.0, y=0.0),
        facing="right",
        current_state="Idle",
        combo_count=my_combo_count,
        blockstun=0,
        hitlag=0,
        state_interruptable=True,
        can_feint=False,
        grounded=True,
    )
    opp = FighterObservation(
        id=opp_id,
        character="Cowboy",
        hp=opp_hp,
        max_hp=max_hp,
        meter=opp_meter,
        burst=opp_burst,
        position=Vector2(x=opp_position_x, y=opp_position_y),
        velocity=Vector2(x=0.0, y=0.0),
        facing="left",
        current_state="Idle",
        combo_count=opp_combo_count,
        blockstun=opp_blockstun,
        hitlag=0,
        state_interruptable=True,
        can_feint=False,
        grounded=True,
    )

    fighters = (me, opp) if player_id == "p1" else (opp, me)

    obs = Observation(
        tick=tick,
        frame=0,
        active_player=player_id,
        fighters=fighters,
        objects=(),
        stage={"width": stage_width},
        history=(),
    )

    wait_action = LegalAction(
        action="Wait",
        payload_spec={},
        supports=LegalActionSupports(di=True, feint=False, reverse=False),
    )

    return DecisionRequest(
        match_id="test-match",
        turn_id=turn_id,
        player_id=player_id,
        deadline_ms=10000,
        state_hash="abc123",
        legal_actions_hash="def456",
        decision_type=DecisionType.TURN_ACTION,
        observation=obs,
        legal_actions=(wait_action,),
    )


def make_decision(
    *,
    action: str = "Wait",
    turn_id: int = 1,
    was_fallback: bool = False,
    latency_ms: int | None = None,
) -> ActionDecision:
    """Create an ActionDecision with sensible defaults."""
    return ActionDecision(
        match_id="test-match",
        turn_id=turn_id,
        action=action,
        data=None,
        extra=DecisionExtras(di=None, feint=False, reverse=False),
        policy_id="test-policy",
        latency_ms=latency_ms,
        fallback_reason="timeout" if was_fallback else None,
    )


def make_match_ended(
    *,
    winner: str | None = "p1",
    end_reason: str = "ko",
    total_turns: int = 30,
) -> MatchEnded:
    """Create a MatchEnded with sensible defaults."""
    return MatchEnded(
        match_id="test-match",
        winner=winner,
        end_reason=end_reason,
        total_turns=total_turns,
        end_tick=300,
        end_frame=0,
    )
