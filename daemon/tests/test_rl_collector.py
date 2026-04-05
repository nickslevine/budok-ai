"""Tests for the trajectory collector."""

from __future__ import annotations

from yomi_daemon.rl.collector import TrajectoryCollector
from yomi_daemon.rl.rewards import RewardConfig
from yomi_daemon.rl.trajectory import MatchOutcomeType

from tests.rl_fixtures import make_decision, make_match_ended, make_request


class TestTrajectoryCollector:
    def test_collects_single_step(self) -> None:
        collector = TrajectoryCollector(
            match_id="m1",
            player_id="p1",
            reward_config=RewardConfig.sparse(),
        )
        request = make_request(player_id="p1")
        decision = make_decision(action="HSlash2")
        collector.record_decision(request, decision)

        ended = make_match_ended(winner="p1")
        traj = collector.finalize(ended)

        assert len(traj) == 1
        assert traj.steps[0].action == "HSlash2"
        assert traj.outcome is not None
        assert traj.outcome.outcome == MatchOutcomeType.WIN

    def test_collects_multiple_steps(self) -> None:
        collector = TrajectoryCollector(
            match_id="m1",
            player_id="p1",
            reward_config=RewardConfig.sparse(),
        )

        for turn_id in range(1, 4):
            request = make_request(player_id="p1", turn_id=turn_id)
            decision = make_decision(action="Wait", turn_id=turn_id)
            collector.record_decision(request, decision)

        ended = make_match_ended(winner="p2")
        traj = collector.finalize(ended)

        assert len(traj) == 3
        assert traj.outcome is not None
        assert traj.outcome.outcome == MatchOutcomeType.LOSS

    def test_sets_character_info(self) -> None:
        collector = TrajectoryCollector(
            match_id="m1",
            player_id="p1",
            reward_config=RewardConfig.sparse(),
        )
        request = make_request(player_id="p1")
        decision = make_decision()
        collector.record_decision(request, decision)

        ended = make_match_ended()
        traj = collector.finalize(ended)

        assert traj.character == "Ninja"
        assert traj.opponent_character == "Cowboy"

    def test_terminal_reward_on_last_step(self) -> None:
        collector = TrajectoryCollector(
            match_id="m1",
            player_id="p1",
            reward_config=RewardConfig.sparse(),  # win/loss only
        )

        request = make_request(player_id="p1")
        decision = make_decision()
        collector.record_decision(request, decision)

        ended = make_match_ended(winner="p1")
        traj = collector.finalize(ended)

        # Sparse config: step rewards are 0, terminal is +1
        assert traj.rewards[-1] == 1.0
