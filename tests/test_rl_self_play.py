from __future__ import annotations

import numpy as np

from scripts.rl_self_play import (
    RolloutStep,
    is_meaningful_step,
    scale_advantages,
    score_player_steps,
    should_run_periodic_eval,
)


def make_step(
    *,
    player_id: str = "p1",
    turn_id: int = 1,
    my_hp: int = 100,
    opp_hp: int = 100,
    legal_actions_count: int = 2,
    was_fallback: bool = False,
) -> RolloutStep:
    return RolloutStep(
        prompt_text="prompt",
        action_json='{"action":"Wait"}',
        action_name="Wait",
        player_id=player_id,
        turn_id=turn_id,
        my_hp=my_hp,
        opp_hp=opp_hp,
        max_hp=100,
        was_fallback=was_fallback,
        legal_actions_count=legal_actions_count,
    )


def test_is_meaningful_step_filters_fallbacks_and_single_action_turns() -> None:
    assert is_meaningful_step(make_step(legal_actions_count=2)) is True
    assert is_meaningful_step(make_step(legal_actions_count=1)) is False
    assert (
        is_meaningful_step(make_step(was_fallback=True, legal_actions_count=5)) is False
    )


def test_score_player_steps_uses_last_meaningful_step_for_terminal_reward_and_rtg() -> (
    None
):
    steps = [
        make_step(turn_id=1, my_hp=100, opp_hp=100, legal_actions_count=3),
        make_step(turn_id=2, my_hp=100, opp_hp=100, legal_actions_count=1),
        make_step(turn_id=3, my_hp=90, opp_hp=70, legal_actions_count=4),
    ]

    scored = score_player_steps(
        steps,
        player_id="p1",
        winner="p1",
        hp_delta_weight=1.0,
        terminal_reward=1.0,
        gamma=0.5,
    )

    assert [entry.step.turn_id for entry in scored] == [1, 3]
    assert scored[0].dense_reward == 0.0
    assert scored[0].terminal_reward == 0.0
    assert scored[0].reward == 0.0
    assert scored[0].return_to_go == 0.6

    # Damage dealt 30, damage taken 10 => +0.2 dense reward, plus +1 terminal.
    assert scored[1].dense_reward == 0.2
    assert scored[1].terminal_reward == 1.0
    assert scored[1].reward == 1.2
    assert scored[1].return_to_go == 1.2


def test_scale_advantages_uses_robust_percentile_and_reports_clip_fraction() -> None:
    returns = np.array([0.0, 1.0, 2.0, 100.0], dtype=np.float32)

    advantages, clip_fraction = scale_advantages(
        returns,
        scale_percentile=90.0,
        clip=1.0,
    )

    assert advantages.shape == returns.shape
    assert clip_fraction > 0.0
    assert np.all(np.abs(advantages) <= 1.0 + 1e-6)


def test_should_run_periodic_eval_includes_cadence_and_final_iteration() -> None:
    assert (
        should_run_periodic_eval(iteration_index=0, num_iterations=5, eval_every=0)
        is False
    )
    assert (
        should_run_periodic_eval(iteration_index=0, num_iterations=5, eval_every=2)
        is False
    )
    assert (
        should_run_periodic_eval(iteration_index=1, num_iterations=5, eval_every=2)
        is True
    )
    assert (
        should_run_periodic_eval(iteration_index=3, num_iterations=5, eval_every=2)
        is True
    )
    assert (
        should_run_periodic_eval(iteration_index=4, num_iterations=5, eval_every=2)
        is True
    )
