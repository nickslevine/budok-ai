"""Serialization for trajectories and training data.

Handles reading/writing trajectory data in JSONL format for offline
training with Tinker or other RL frameworks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from yomi_daemon.rl.trajectory import EpisodeResult, Trajectory


def trajectory_to_jsonl_records(trajectory: Trajectory) -> list[dict[str, Any]]:
    """Convert a trajectory into a list of JSONL-serializable records.

    Each record is one step: the prompt (serialized observation + legal actions)
    paired with the chosen action and its reward. This is the format Tinker's
    GRPO loop consumes -- each record is an independent (prompt, completion, reward)
    tuple.
    """
    records: list[dict[str, Any]] = []
    for i, (step, reward) in enumerate(zip(trajectory.steps, trajectory.rewards)):
        record: dict[str, Any] = {
            "match_id": trajectory.match_id,
            "player_id": trajectory.player_id,
            "step_index": i,
            "turn_id": step.turn_id,
            "tick": step.tick,
            "character": trajectory.character,
            "opponent_character": trajectory.opponent_character,
            # Observation: the full request payload for prompt rendering
            "observation": step.request.observation.to_dict(),
            "legal_actions": [la.to_dict() for la in step.request.legal_actions],
            "state_hash": step.request.state_hash,
            # Decision
            "action": step.action,
            "decision_data": step.decision.data,
            "decision_extra": step.decision.extra.to_dict(),
            "was_fallback": step.was_fallback,
            # Reward
            "reward": reward,
            # Context
            "my_hp": step.my_hp,
            "opp_hp": step.opp_hp,
            "my_hp_delta": step.my_hp_delta,
            "opp_hp_delta": step.opp_hp_delta,
        }
        records.append(record)
    return records


def write_trajectories(path: Path, trajectories: list[Trajectory]) -> None:
    """Write trajectories to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for traj in trajectories:
            records = trajectory_to_jsonl_records(traj)
            for record in records:
                f.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def write_episode_summary(path: Path, results: list[EpisodeResult]) -> None:
    """Write episode summaries to a JSON file for training metrics."""
    summaries = []
    for result in results:
        summaries.append(
            {
                "outcome": result.outcome.outcome.value,
                "end_reason": result.outcome.end_reason,
                "total_turns": result.outcome.total_turns,
                "my_final_hp": result.my_final_hp,
                "opp_final_hp": result.opp_final_hp,
                "total_reward": result.total_reward,
                "won": result.outcome.won,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summaries, f, indent=2)


def load_trajectory_records(path: Path) -> list[dict[str, Any]]:
    """Load trajectory records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
