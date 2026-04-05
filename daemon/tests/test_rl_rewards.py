"""Tests for RL reward components and composition."""

from __future__ import annotations

from yomi_daemon.rl.rewards import (
    RewardConfig,
    center_control,
    combo_reward,
    compute_step_reward,
    compute_terminal_reward,
    fallback_penalty,
    hp_delta,
    pressure_reward,
    win_loss,
)
from yomi_daemon.rl.trajectory import MatchOutcome, MatchOutcomeType

from tests.rl_fixtures import make_step


class TestHpDelta:
    def test_positive_when_dealing_damage(self) -> None:
        step = make_step(my_hp_delta=0, opp_hp_delta=-30, max_hp=100)
        assert hp_delta(step) > 0

    def test_negative_when_taking_damage(self) -> None:
        step = make_step(my_hp_delta=-30, opp_hp_delta=0, max_hp=100)
        assert hp_delta(step) < 0

    def test_zero_when_no_delta(self) -> None:
        step = make_step(my_hp_delta=0, opp_hp_delta=0, max_hp=100)
        assert hp_delta(step) == 0.0

    def test_zero_when_delta_missing(self) -> None:
        step = make_step(my_hp_delta=None, opp_hp_delta=None, max_hp=100)
        assert hp_delta(step) == 0.0

    def test_normalized_to_max_hp(self) -> None:
        step = make_step(my_hp_delta=0, opp_hp_delta=-50, max_hp=100)
        assert hp_delta(step) == 0.5

        step200 = make_step(my_hp_delta=0, opp_hp_delta=-50, max_hp=200)
        assert hp_delta(step200) == 0.25

    def test_net_when_both_take_damage(self) -> None:
        # We take 10, opponent takes 30. Net = 30 - 10 = 20.
        step = make_step(my_hp_delta=-10, opp_hp_delta=-30, max_hp=100)
        assert hp_delta(step) == 0.2


class TestFallbackPenalty:
    def test_penalty_on_fallback(self) -> None:
        step = make_step(was_fallback=True)
        assert fallback_penalty(step) == -1.0

    def test_no_penalty_normally(self) -> None:
        step = make_step(was_fallback=False)
        assert fallback_penalty(step) == 0.0


class TestComboReward:
    def test_positive_when_comboing(self) -> None:
        step = make_step(my_combo_count=3, opp_combo_count=0)
        assert combo_reward(step) == 0.3

    def test_negative_when_being_comboed(self) -> None:
        step = make_step(my_combo_count=0, opp_combo_count=5)
        assert combo_reward(step) == -0.5


class TestCenterControl:
    def test_max_at_center(self) -> None:
        step = make_step(my_position_x=0.0, stage_width=1100.0)
        assert center_control(step) == 1.0

    def test_zero_at_edge(self) -> None:
        step = make_step(my_position_x=550.0, stage_width=1100.0)
        assert center_control(step) == 0.0

    def test_midway(self) -> None:
        step = make_step(my_position_x=275.0, stage_width=1100.0)
        assert center_control(step) == 0.5


class TestPressureReward:
    def test_capped_at_one(self) -> None:
        step = make_step(opp_blockstun=10)
        assert pressure_reward(step) == 1.0

    def test_zero_when_no_blockstun(self) -> None:
        step = make_step(opp_blockstun=0)
        assert pressure_reward(step) == 0.0


class TestWinLoss:
    def test_win(self) -> None:
        outcome = MatchOutcome(
            outcome=MatchOutcomeType.WIN,
            end_reason="ko",
            total_turns=30,
            my_final_hp=50,
            opp_final_hp=0,
        )
        assert win_loss(outcome) == 1.0

    def test_loss(self) -> None:
        outcome = MatchOutcome(
            outcome=MatchOutcomeType.LOSS,
            end_reason="ko",
            total_turns=30,
            my_final_hp=0,
            opp_final_hp=50,
        )
        assert win_loss(outcome) == -1.0

    def test_draw(self) -> None:
        outcome = MatchOutcome(
            outcome=MatchOutcomeType.DRAW,
            end_reason="timeout",
            total_turns=60,
            my_final_hp=50,
            opp_final_hp=50,
        )
        assert win_loss(outcome) == 0.0


class TestRewardConfig:
    def test_default_config(self) -> None:
        config = RewardConfig.default()
        assert "hp_delta" in config.components
        assert "fallback" in config.components

    def test_sparse_config_has_no_components(self) -> None:
        config = RewardConfig.sparse()
        assert len(config.components) == 0

    def test_compute_step_reward(self) -> None:
        config = RewardConfig(
            components={"hp_delta": (hp_delta, 1.0)},
        )
        step = make_step(my_hp_delta=0, opp_hp_delta=-50, max_hp=100)
        result = compute_step_reward(step, config)
        assert result.total == 0.5
        assert result.components["hp_delta"] == 0.5

    def test_compute_terminal_reward(self) -> None:
        config = RewardConfig(terminal_weight=2.0)
        outcome = MatchOutcome(
            outcome=MatchOutcomeType.WIN,
            end_reason="ko",
            total_turns=30,
            my_final_hp=50,
            opp_final_hp=0,
        )
        assert compute_terminal_reward(outcome, config) == 2.0

    def test_weighted_combination(self) -> None:
        config = RewardConfig(
            components={
                "hp_delta": (hp_delta, 0.5),
                "fallback": (fallback_penalty, 0.1),
            },
        )
        step = make_step(my_hp_delta=0, opp_hp_delta=-100, max_hp=100, was_fallback=True)
        result = compute_step_reward(step, config)
        # hp_delta = 1.0 * 0.5 = 0.5, fallback = -1.0 * 0.1 = -0.1
        assert result.total == 0.4
