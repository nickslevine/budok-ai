"""Trajectory collector that hooks into the daemon's decision pipeline.

The TrajectoryCollector observes request/decision pairs as they flow
through the daemon and assembles them into Trajectory objects for
offline RL training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from yomi_daemon.protocol import (
    ActionDecision,
    DecisionRequest,
    HistoryEntry,
    MatchEnded,
)
from yomi_daemon.rl.rewards import RewardConfig, compute_step_reward, compute_terminal_reward
from yomi_daemon.rl.trajectory import MatchOutcome, Step, Trajectory


@dataclass(slots=True)
class TrajectoryCollector:
    """Collects Steps into Trajectories during live matches.

    One collector per player per match. Feed it request/decision pairs
    as they happen, then finalize when the match ends.
    """

    match_id: str
    player_id: str
    reward_config: RewardConfig = field(default_factory=RewardConfig.default)

    _trajectory: Trajectory = field(init=False)
    _pending_step: Step | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._trajectory = Trajectory(
            match_id=self.match_id,
            player_id=self.player_id,
        )

    def record_decision(
        self,
        request: DecisionRequest,
        decision: ActionDecision,
    ) -> None:
        """Record a request/decision pair.

        The outcome for the *previous* step is extracted from this
        request's observation history (the most recent entry reflects
        what happened after the previous action).
        """
        # Resolve outcome for the previous pending step
        outcome_entry = _latest_history_entry(request.observation.history)
        if self._pending_step is not None and outcome_entry is not None:
            # Rebuild the previous step with outcome data
            enriched = Step.from_request_decision(
                self._pending_step.request,
                self._pending_step.decision,
                outcome_entry=outcome_entry,
            )
            reward = compute_step_reward(enriched, self.reward_config)
            self._trajectory.add_step(enriched, reward.total)
        elif self._pending_step is not None:
            # No outcome data available -- record with zero reward
            reward = compute_step_reward(self._pending_step, self.reward_config)
            self._trajectory.add_step(self._pending_step, reward.total)

        # Set character info on first step
        if not self._trajectory.character:
            obs = request.observation
            for f in obs.fighters:
                if f.id == self.player_id:
                    self._trajectory.character = f.character
                else:
                    self._trajectory.opponent_character = f.character

        # Current step becomes pending (will be enriched next turn)
        self._pending_step = Step.from_request_decision(request, decision)

    def finalize(self, match_ended: MatchEnded) -> Trajectory:
        """Finalize the trajectory with the match result.

        Returns the completed Trajectory with all rewards computed.
        """
        # Flush the last pending step
        if self._pending_step is not None:
            reward = compute_step_reward(self._pending_step, self.reward_config)
            self._trajectory.add_step(self._pending_step, reward.total)
            self._pending_step = None

        outcome = MatchOutcome.from_match_ended(match_ended, self.player_id)
        terminal = compute_terminal_reward(outcome, self.reward_config)
        self._trajectory.finalize(match_ended, terminal_reward=terminal)

        return self._trajectory

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory


def _latest_history_entry(history: tuple[HistoryEntry, ...]) -> HistoryEntry | None:
    """Get the most recent history entry with outcome data."""
    for entry in reversed(history):
        if entry.p1_hp_delta is not None or entry.p2_hp_delta is not None:
            return entry
    return None
