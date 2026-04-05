"""Trajectory data structures for collecting RL rollout data.

A Trajectory is a sequence of Steps from a single match, played from one
player's perspective. Steps are extracted from the existing protocol types
(DecisionRequest, ActionDecision, HistoryEntry) with fields pre-resolved
for reward computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from yomi_daemon.protocol import (
    ActionDecision,
    DecisionRequest,
    FighterObservation,
    HistoryEntry,
    MatchEnded,
    Observation,
)


class MatchOutcomeType(StrEnum):
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


# ---------------------------------------------------------------------------
# Step: a single decision point with context for reward computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Step:
    """One decision point from a single player's perspective.

    Fields are pre-resolved from the protocol types so reward functions
    don't need to navigate the observation tree.
    """

    turn_id: int
    tick: int

    # The raw protocol objects (for prompt rendering, serialization, etc.)
    request: DecisionRequest
    decision: ActionDecision

    # Pre-resolved fighter state (from our perspective)
    my_hp: int
    opp_hp: int
    max_hp: int
    my_position_x: float
    my_position_y: float
    opp_position_x: float
    opp_position_y: float
    my_combo_count: int
    opp_combo_count: int
    opp_blockstun: int
    my_meter: int
    opp_meter: int
    my_burst: int
    opp_burst: int

    # Stage info
    stage_width: float

    # Outcome from this turn (from history of the *next* turn, or post-match)
    my_hp_delta: int | None = None
    opp_hp_delta: int | None = None
    my_outcome: str | None = None
    opp_outcome: str | None = None

    # Policy metadata
    was_fallback: bool = False
    action: str = ""
    latency_ms: int | None = None

    @staticmethod
    def from_request_decision(
        request: DecisionRequest,
        decision: ActionDecision,
        *,
        outcome_entry: HistoryEntry | None = None,
    ) -> Step:
        """Build a Step from a request/decision pair.

        Args:
            request: The DecisionRequest for this turn.
            decision: The ActionDecision the policy produced.
            outcome_entry: The HistoryEntry from the *next* turn that
                contains the HP deltas and outcomes for this turn.
                None if this is the last turn or data is unavailable.
        """
        obs = request.observation
        player_id = request.player_id
        me, opp = _resolve_fighters(obs, player_id)

        stage_width = 0.0
        raw_width = obs.stage.get("width")
        if isinstance(raw_width, int | float):
            stage_width = float(raw_width)

        my_hp_delta: int | None = None
        opp_hp_delta: int | None = None
        my_outcome: str | None = None
        opp_outcome: str | None = None

        if outcome_entry is not None:
            if player_id == "p1":
                my_hp_delta = outcome_entry.p1_hp_delta
                opp_hp_delta = outcome_entry.p2_hp_delta
                my_outcome = outcome_entry.p1_outcome
                opp_outcome = outcome_entry.p2_outcome
            else:
                my_hp_delta = outcome_entry.p2_hp_delta
                opp_hp_delta = outcome_entry.p1_hp_delta
                my_outcome = outcome_entry.p2_outcome
                opp_outcome = outcome_entry.p1_outcome

        return Step(
            turn_id=request.turn_id,
            tick=obs.tick,
            request=request,
            decision=decision,
            my_hp=me.hp,
            opp_hp=opp.hp,
            max_hp=me.max_hp,
            my_position_x=me.position.x,
            my_position_y=me.position.y,
            opp_position_x=opp.position.x,
            opp_position_y=opp.position.y,
            my_combo_count=me.combo_count,
            opp_combo_count=opp.combo_count,
            opp_blockstun=opp.blockstun,
            my_meter=me.meter,
            opp_meter=opp.meter,
            my_burst=me.burst,
            opp_burst=opp.burst,
            stage_width=stage_width,
            my_hp_delta=my_hp_delta,
            opp_hp_delta=opp_hp_delta,
            my_outcome=my_outcome,
            opp_outcome=opp_outcome,
            was_fallback=decision.fallback_reason is not None,
            action=decision.action,
            latency_ms=decision.latency_ms,
        )


# ---------------------------------------------------------------------------
# Match outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """Terminal match result from one player's perspective."""

    outcome: MatchOutcomeType
    end_reason: str
    total_turns: int
    my_final_hp: int
    opp_final_hp: int

    @property
    def won(self) -> bool | None:
        if self.outcome == MatchOutcomeType.WIN:
            return True
        if self.outcome == MatchOutcomeType.LOSS:
            return False
        return None

    @staticmethod
    def from_match_ended(match_ended: MatchEnded, player_id: str) -> MatchOutcome:
        """Derive outcome from the protocol MatchEnded message."""
        if match_ended.winner is None:
            outcome = MatchOutcomeType.DRAW
        elif match_ended.winner == player_id:
            outcome = MatchOutcomeType.WIN
        else:
            outcome = MatchOutcomeType.LOSS

        return MatchOutcome(
            outcome=outcome,
            end_reason=match_ended.end_reason,
            total_turns=match_ended.total_turns,
            # Final HP filled in from last trajectory step if available
            my_final_hp=0,
            opp_final_hp=0,
        )


# ---------------------------------------------------------------------------
# Episode result (match outcome + optional final HP)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Full match result combining outcome with final state."""

    outcome: MatchOutcome
    my_final_hp: int
    opp_final_hp: int
    total_reward: float

    @staticmethod
    def from_trajectory(trajectory: Trajectory) -> EpisodeResult:
        """Compute episode result from a completed trajectory."""
        if trajectory.outcome is None:
            raise ValueError("Cannot compute EpisodeResult from incomplete trajectory")

        my_final_hp = trajectory.steps[-1].my_hp if trajectory.steps else 0
        opp_final_hp = trajectory.steps[-1].opp_hp if trajectory.steps else 0

        # Apply any HP deltas from the last step
        last = trajectory.steps[-1] if trajectory.steps else None
        if last is not None and last.my_hp_delta is not None:
            my_final_hp += last.my_hp_delta
        if last is not None and last.opp_hp_delta is not None:
            opp_final_hp += last.opp_hp_delta

        return EpisodeResult(
            outcome=trajectory.outcome,
            my_final_hp=max(0, my_final_hp),
            opp_final_hp=max(0, opp_final_hp),
            total_reward=sum(trajectory.rewards),
        )


# ---------------------------------------------------------------------------
# Trajectory: full episode of steps
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Trajectory:
    """A sequence of Steps from one player's perspective across a match.

    Mutable during collection, frozen after finalization.
    """

    match_id: str
    player_id: str
    character: str = ""
    opponent_character: str = ""

    steps: list[Step] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    outcome: MatchOutcome | None = None

    def add_step(self, step: Step, reward: float = 0.0) -> None:
        """Append a step and its associated reward."""
        self.steps.append(step)
        self.rewards.append(reward)

    def finalize(self, match_ended: MatchEnded, terminal_reward: float = 0.0) -> None:
        """Mark the trajectory as complete with the match outcome."""
        self.outcome = MatchOutcome.from_match_ended(match_ended, self.player_id)
        if self.steps:
            self.rewards[-1] += terminal_reward

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def __len__(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_fighters(
    obs: Observation, player_id: str
) -> tuple[FighterObservation, FighterObservation]:
    """Return (my_fighter, opponent_fighter) given a player_id."""
    fighters = obs.fighters
    if len(fighters) != 2:
        raise ValueError(f"Expected 2 fighters, got {len(fighters)}")
    if fighters[0].id == player_id:
        return fighters[0], fighters[1]
    if fighters[1].id == player_id:
        return fighters[1], fighters[0]
    raise ValueError(f"Player {player_id!r} not found in fighters: {[f.id for f in fighters]}")
