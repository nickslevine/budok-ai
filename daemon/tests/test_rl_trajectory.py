"""Tests for RL trajectory data structures."""

from __future__ import annotations

import pytest

from yomi_daemon.rl.trajectory import (
    EpisodeResult,
    MatchOutcome,
    MatchOutcomeType,
    Step,
    Trajectory,
)

from tests.rl_fixtures import make_decision, make_match_ended, make_request, make_step


class TestStepFromRequestDecision:
    def test_resolves_fighters_correctly(self) -> None:
        request = make_request(player_id="p1", my_hp=80, opp_hp=60)
        decision = make_decision(action="HSlash2")
        step = Step.from_request_decision(request, decision)

        assert step.my_hp == 80
        assert step.opp_hp == 60
        assert step.action == "HSlash2"

    def test_resolves_p2_perspective(self) -> None:
        request = make_request(player_id="p2", my_hp=70, opp_hp=90)
        decision = make_decision()
        step = Step.from_request_decision(request, decision)

        assert step.my_hp == 70
        assert step.opp_hp == 90

    def test_includes_stage_width(self) -> None:
        request = make_request(stage_width=1200.0)
        decision = make_decision()
        step = Step.from_request_decision(request, decision)

        assert step.stage_width == 1200.0

    def test_fallback_detection(self) -> None:
        request = make_request()
        decision = make_decision(was_fallback=True)
        step = Step.from_request_decision(request, decision)

        assert step.was_fallback is True


class TestMatchOutcome:
    def test_win(self) -> None:
        ended = make_match_ended(winner="p1")
        outcome = MatchOutcome.from_match_ended(ended, "p1")
        assert outcome.outcome == MatchOutcomeType.WIN
        assert outcome.won is True

    def test_loss(self) -> None:
        ended = make_match_ended(winner="p2")
        outcome = MatchOutcome.from_match_ended(ended, "p1")
        assert outcome.outcome == MatchOutcomeType.LOSS
        assert outcome.won is False

    def test_draw(self) -> None:
        ended = make_match_ended(winner=None)
        outcome = MatchOutcome.from_match_ended(ended, "p1")
        assert outcome.outcome == MatchOutcomeType.DRAW
        assert outcome.won is None


class TestTrajectory:
    def test_add_step(self) -> None:
        traj = Trajectory(match_id="m1", player_id="p1")
        step = make_step()
        traj.add_step(step, reward=0.5)

        assert len(traj) == 1
        assert traj.total_reward == 0.5

    def test_finalize(self) -> None:
        traj = Trajectory(match_id="m1", player_id="p1")
        traj.add_step(make_step(), reward=0.3)
        traj.add_step(make_step(turn_id=2), reward=0.2)

        ended = make_match_ended(winner="p1")
        traj.finalize(ended, terminal_reward=1.0)

        assert traj.outcome is not None
        assert traj.outcome.outcome == MatchOutcomeType.WIN
        # Terminal reward added to last step
        assert traj.rewards[-1] == 1.2  # 0.2 + 1.0
        assert traj.total_reward == 1.5  # 0.3 + 0.2 + 1.0

    def test_finalize_empty_trajectory(self) -> None:
        traj = Trajectory(match_id="m1", player_id="p1")
        ended = make_match_ended(winner=None)
        # Should not raise even with no steps
        traj.finalize(ended, terminal_reward=0.0)
        assert traj.outcome is not None


class TestEpisodeResult:
    def test_from_trajectory(self) -> None:
        traj = Trajectory(match_id="m1", player_id="p1")
        traj.add_step(make_step(my_hp=100, opp_hp=100), reward=0.3)
        traj.add_step(make_step(my_hp=80, opp_hp=50, turn_id=2), reward=0.5)

        ended = make_match_ended(winner="p1")
        traj.finalize(ended, terminal_reward=1.0)

        result = EpisodeResult.from_trajectory(traj)
        assert result.outcome.won is True
        assert result.total_reward == 1.8  # 0.3 + 0.5 + 1.0

    def test_raises_on_incomplete(self) -> None:
        traj = Trajectory(match_id="m1", player_id="p1")
        traj.add_step(make_step(), reward=0.0)

        with pytest.raises(ValueError, match="incomplete"):
            EpisodeResult.from_trajectory(traj)
