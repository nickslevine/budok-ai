"""Utilities for evaluating candidate policies against RL v3 benchmark suites."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from yomi_daemon.rl.v3 import (
    PolicySpec,
    SuiteConfig,
    SuiteSeedBank,
    aggregate_suite_score,
    build_head_to_head_config,
    clone_policy,
    count_completed_runs,
    default_policy_for_model,
    resolve_suite_seed_bank,
    summarize_runs_root,
    suite_entry_sides,
    timestamp_utc,
)


def run_match_subprocess(
    *,
    repo_root: Path,
    config_path: Path,
    runs_dir: Path,
    trace_seed: int,
    timeout_s: int,
    skip_mod_push: bool = True,
) -> bool:
    cmd = [
        str(repo_root / "scripts" / "run_match.sh"),
        "--daemon-config",
        str(config_path),
        "--runs-root",
        str(runs_dir),
        "--trace-seed",
        str(trace_seed),
        "--no-replay",
    ]
    if skip_mod_push:
        cmd.append("--skip-mod-push")
    try:
        result = subprocess.run(cmd, cwd=repo_root, timeout=timeout_s, capture_output=False)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    Match timed out after {timeout_s}s")
        return False


def write_temp_config(
    base_template: dict[str, Any],
    *,
    p1_policy: PolicySpec,
    p2_policy: PolicySpec,
) -> Path:
    document = build_head_to_head_config(base_template, p1_policy=p1_policy, p2_policy=p2_policy)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="rl_suite_eval_")
    import os as _os

    with _os.fdopen(fd, "w") as f:
        json.dump(document, f, indent=2)
    return Path(path)


def _is_tinker_weights_checkpoint(model: str) -> bool:
    return model.startswith("tinker://") and "/weights/" in model and "/sampler_weights/" not in model


async def _export_sampler_from_state_async(*, state_path: str, sampler_name: str) -> str:
    import tinker

    service = tinker.ServiceClient()
    training_client = await service.create_training_client_from_state_async(path=state_path)
    sampler_future = await training_client.save_weights_for_sampler_async(name=sampler_name)
    sampler_result = await sampler_future.result_async()
    return sampler_result.path


def resolve_model_for_evaluation(model: str, *, sampler_name: str) -> str:
    if not _is_tinker_weights_checkpoint(model):
        return model
    print(
        f"Model {model} is a persistent weights checkpoint; exporting sampler '{sampler_name}' for evaluation..."
    )
    sampler_path = asyncio.run(
        _export_sampler_from_state_async(state_path=model, sampler_name=sampler_name)
    )
    print(f"  Exported sampler: {sampler_path}")
    return sampler_path


def build_policy_from_cli(
    *,
    policy_id: str,
    model: str,
    prompt_version: str,
    temperature: float,
    max_tokens: int,
    label: str | None,
    provider: str | None = None,
    credential_env_var: str | None = None,
    base_url: str | None = None,
) -> PolicySpec:
    base = default_policy_for_model(
        policy_id=policy_id,
        model=model,
        prompt_version=prompt_version,
        temperature=temperature,
        max_tokens=max_tokens,
        label=label,
    )
    options = dict(base.options)
    if base_url is not None:
        options["base_url"] = base_url
    resolved_provider = provider or base.provider
    resolved_credential = credential_env_var or base.credential_env_var
    if (
        resolved_provider != base.provider
        or resolved_credential != base.credential_env_var
        or options != base.options
    ):
        return PolicySpec(
            policy_id=base.policy_id,
            provider=resolved_provider,
            model=base.model,
            prompt_version=base.prompt_version,
            credential_env_var=resolved_credential,
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            options=options,
            label=base.label,
        )
    return base


def combine_side_summaries(by_side: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_games = sum(int(summary.get("games", 0)) for summary in by_side.values())
    wins = sum(int(summary.get("wins", 0)) for summary in by_side.values())
    losses = sum(int(summary.get("losses", 0)) for summary in by_side.values())
    draws = sum(int(summary.get("draws", 0)) for summary in by_side.values())
    fallback_count = sum(int(summary.get("candidate_fallbacks", 0)) for summary in by_side.values())
    decision_count = sum(int(summary.get("candidate_decisions", 0)) for summary in by_side.values())
    hp_sum = sum(
        float(summary.get("avg_hp_diff", 0.0)) * int(summary.get("games", 0))
        for summary in by_side.values()
    )
    turn_sum = sum(
        float(summary.get("avg_turns", 0.0)) * int(summary.get("games", 0))
        for summary in by_side.values()
    )
    action_histogram: Counter[str] = Counter()
    opening_histogram: Counter[str] = Counter()
    for summary in by_side.values():
        actions = summary.get("candidate_action_histogram", {})
        if isinstance(actions, dict):
            for action, count in actions.items():
                if isinstance(action, str) and isinstance(count, int | float):
                    action_histogram[action] += int(count)
        openings = summary.get("candidate_opening_histogram", {})
        if isinstance(openings, dict):
            for action, count in openings.items():
                if isinstance(action, str) and isinstance(count, int | float):
                    opening_histogram[action] += int(count)
    unique_actions = len(action_histogram)
    unique_openings = len(opening_histogram)
    return {
        "games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": (wins / total_games) if total_games else 0.0,
        "avg_hp_diff": round(hp_sum / total_games, 1) if total_games else 0.0,
        "avg_turns": round(turn_sum / total_games, 1) if total_games else 0.0,
        "candidate_fallbacks": fallback_count,
        "candidate_decisions": decision_count,
        "candidate_fallback_rate": (fallback_count / decision_count) if decision_count else 0.0,
        "candidate_action_histogram": dict(action_histogram),
        "candidate_opening_histogram": dict(opening_histogram),
        "candidate_unique_actions": unique_actions,
        "candidate_unique_openings": unique_openings,
        "candidate_action_diversity": (unique_actions / decision_count) if decision_count else 0.0,
        "candidate_opening_diversity": (unique_openings / total_games) if total_games else 0.0,
    }


def evaluate_suite(
    *,
    repo_root: Path,
    candidate_policy: PolicySpec,
    suite: SuiteConfig,
    output_root: Path,
    base_template_path: Path,
    match_timeout_s: int,
    seed_base: int,
    seed_bank: SuiteSeedBank | None = None,
    match_max_attempts: int = 1,
    skip_mod_push: bool = True,
) -> dict[str, Any]:
    base_template = json.loads(base_template_path.read_text())
    resolved_seed_bank = resolve_suite_seed_bank(suite, seed_base=seed_base, seed_bank=seed_bank)
    seed_entries = resolved_seed_bank.entry_map()
    entries_result: list[dict[str, Any]] = []
    total_completed = 0
    total_failed = 0

    for entry in suite.entries:
        print(f"\n[{entry.entry_id}] vs {entry.label or entry.opponent.policy_id}")
        sides = list(suite_entry_sides(entry))
        entry_seed_bank = seed_entries[entry.entry_id]
        by_side: dict[str, dict[str, Any]] = {}
        entry_completed = 0
        entry_failed = 0

        for side in sides:
            side_root = output_root / entry.entry_id / side
            side_root.mkdir(parents=True, exist_ok=True)
            candidate_side = clone_policy(
                candidate_policy,
                policy_id=f"{candidate_policy.policy_id}-{side}",
            )
            opponent_side = clone_policy(
                entry.opponent,
                policy_id=f"{entry.opponent.policy_id}-{side}",
            )
            if side == "p1":
                p1_policy = candidate_side
                p2_policy = opponent_side
            else:
                p1_policy = opponent_side
                p2_policy = candidate_side

            side_seeds = entry_seed_bank.seeds_by_side[side]
            for match_index, trace_seed in enumerate(side_seeds):
                completed_before = count_completed_runs(side_root)
                match_completed = False
                for attempt_index in range(match_max_attempts):
                    config_path = write_temp_config(
                        base_template,
                        p1_policy=p1_policy,
                        p2_policy=p2_policy,
                    )
                    attempt_label = (
                        f" attempt {attempt_index + 1}/{match_max_attempts}"
                        if match_max_attempts > 1
                        else ""
                    )
                    print(
                        f"  side={side} match {match_index + 1}/{len(side_seeds)} "
                        f"seed={trace_seed}{attempt_label}"
                    )
                    ok = run_match_subprocess(
                        repo_root=repo_root,
                        config_path=config_path,
                        runs_dir=side_root,
                        trace_seed=trace_seed,
                        timeout_s=match_timeout_s,
                        skip_mod_push=skip_mod_push,
                    )
                    config_path.unlink(missing_ok=True)
                    completed_after = count_completed_runs(side_root)
                    if ok and completed_after > completed_before:
                        match_completed = True
                        break
                if match_completed:
                    entry_completed += 1
                    total_completed += 1
                else:
                    entry_failed += 1
                    total_failed += 1

            side_summary = summarize_runs_root(side_root, candidate_player_id=side)
            by_side[side] = side_summary

        aggregate = combine_side_summaries(by_side)
        entries_result.append(
            {
                "entry_id": entry.entry_id,
                "label": entry.label,
                "weight": entry.weight,
                "num_matches": entry.num_matches,
                "sides": entry.sides,
                "opponent": entry.opponent.to_dict(),
                "completed": entry_completed,
                "failed": entry_failed,
                "aggregate": aggregate,
                "by_side": by_side,
                "seeds_by_side": {
                    side_name: list(seeds)
                    for side_name, seeds in entry_seed_bank.seeds_by_side.items()
                },
            }
        )
        print(
            f"  -> win_rate={aggregate['win_rate']:.1%} avg_hp_diff={aggregate['avg_hp_diff']} "
            f"fallback_rate={aggregate['candidate_fallback_rate']:.1%}"
        )

    result: dict[str, Any] = {
        "evaluated_at": timestamp_utc(),
        "candidate": candidate_policy.to_dict(),
        "suite": suite.to_dict(),
        "seed_bank": resolved_seed_bank.to_dict(),
        "entries": entries_result,
        "total_completed": total_completed,
        "total_failed": total_failed,
    }
    result["aggregate_score"] = aggregate_suite_score(result)
    return result
