"""Composable reward functions for YOMI Hustle RL training.

Each reward component is a pure function: (step_context) -> float.
Components are combined via RewardConfig with named weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yomi_daemon.rl.trajectory import MatchOutcome, Step


# ---------------------------------------------------------------------------
# Reward component protocol
# ---------------------------------------------------------------------------

RewardComponent = Callable[["Step"], float]
"""A reward component takes a Step and returns a scalar reward signal."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RewardResult:
    """Breakdown of a reward computation into named components."""

    total: float
    components: dict[str, float]


# ---------------------------------------------------------------------------
# Built-in reward components
# ---------------------------------------------------------------------------


def hp_delta(step: Step) -> float:
    """Normalized HP delta: positive when we deal more damage than we take.

    Returns a value in roughly [-1, 1] based on the net HP change
    relative to max HP. If HP delta data is missing from history,
    returns 0.
    """
    if step.my_hp_delta is None or step.opp_hp_delta is None:
        return 0.0
    if step.max_hp == 0:
        return 0.0
    # opp_hp_delta is negative when opponent takes damage, my_hp_delta
    # is negative when we take damage. Net gain = damage dealt - damage taken.
    net = (-step.opp_hp_delta) - (-step.my_hp_delta)
    return net / step.max_hp


def fallback_penalty(step: Step) -> float:
    """Penalty when the policy timed out or produced an invalid action."""
    return -1.0 if step.was_fallback else 0.0


def combo_reward(step: Step) -> float:
    """Reward for extending combos (positive) or being comboed (negative)."""
    return (step.my_combo_count - step.opp_combo_count) / 10.0


def center_control(step: Step) -> float:
    """Reward for occupying center stage. Range [-1, 1].

    The stage center is at x=0, with the stage extending to roughly
    +/- stage_width/2. Being at center = 1.0, being at the edge = 0.0.
    """
    if step.stage_width <= 0:
        return 0.0
    half_width = step.stage_width / 2.0
    my_dist = abs(step.my_position_x)
    return max(0.0, 1.0 - my_dist / half_width)


def pressure_reward(step: Step) -> float:
    """Reward for putting the opponent in blockstun."""
    return min(step.opp_blockstun / 5.0, 1.0)


# ---------------------------------------------------------------------------
# Terminal reward
# ---------------------------------------------------------------------------


def win_loss(outcome: MatchOutcome) -> float:
    """Terminal reward: +1 for win, -1 for loss, 0 for draw."""
    if outcome.won is True:
        return 1.0
    if outcome.won is False:
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Config and composition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weighted combination of reward components.

    Usage:
        config = RewardConfig(components={
            "hp_delta": (hp_delta, 0.3),
            "fallback": (fallback_penalty, 0.1),
        })
        result = compute_step_reward(step, config)
    """

    components: dict[str, tuple[RewardComponent, float]] = field(default_factory=dict)
    terminal_weight: float = 1.0

    @staticmethod
    def default() -> RewardConfig:
        """Sensible defaults: HP delta + fallback penalty + terminal win/loss."""
        return RewardConfig(
            components={
                "hp_delta": (hp_delta, 0.3),
                "fallback": (fallback_penalty, 0.1),
            },
            terminal_weight=1.0,
        )

    @staticmethod
    def sparse() -> RewardConfig:
        """Win/loss only -- cleanest signal, hardest to learn from."""
        return RewardConfig(components={}, terminal_weight=1.0)

    @staticmethod
    def dense() -> RewardConfig:
        """All available dense signals. Use carefully -- reward hacking risk."""
        return RewardConfig(
            components={
                "hp_delta": (hp_delta, 0.3),
                "fallback": (fallback_penalty, 0.1),
                "combo": (combo_reward, 0.05),
                "center": (center_control, 0.02),
                "pressure": (pressure_reward, 0.03),
            },
            terminal_weight=1.0,
        )


def compute_step_reward(step: Step, config: RewardConfig) -> RewardResult:
    """Compute the weighted reward for a single step."""
    components: dict[str, float] = {}
    total = 0.0
    for name, (fn, weight) in config.components.items():
        raw = fn(step)
        weighted = raw * weight
        components[name] = weighted
        total += weighted
    return RewardResult(total=total, components=components)


def compute_terminal_reward(outcome: MatchOutcome, config: RewardConfig) -> float:
    """Compute the terminal reward for a completed match."""
    return win_loss(outcome) * config.terminal_weight
