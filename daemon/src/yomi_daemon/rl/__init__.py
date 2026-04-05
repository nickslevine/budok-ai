"""RL training primitives for YOMI Hustle LLM agents."""

from yomi_daemon.rl.rewards import (
    RewardComponent,
    RewardConfig,
    RewardResult,
    compute_step_reward,
    compute_terminal_reward,
    fallback_penalty,
    hp_delta,
    win_loss,
)
from yomi_daemon.rl.trajectory import (
    EpisodeResult,
    MatchOutcome,
    Step,
    Trajectory,
)

__all__ = [
    "RewardComponent",
    "RewardConfig",
    "RewardResult",
    "compute_step_reward",
    "compute_terminal_reward",
    "fallback_penalty",
    "hp_delta",
    "win_loss",
    "EpisodeResult",
    "MatchOutcome",
    "Step",
    "Trajectory",
]
