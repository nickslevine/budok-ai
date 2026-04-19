"""RL self-play training on Tinker.

Starts from the SFT v1 checkpoint and improves via self-play + policy
gradient with advantage-scaled cross-entropy. Each iteration:

  1. Save current weights as a Tinker sampler checkpoint
  2. Run N self-play matches (both P1 and P2 use the current policy)
  3. Parse trajectories from runs/, keep only meaningful decisions,
     and compute dense reward + terminal reward + return-to-go
  4. Scale advantages with robust percentile normalization
  5. Tokenize with advantage-scaled weights
  6. Forward/backward with cross_entropy (policy gradient)
  7. Save persistent checkpoints every N iterations

Usage:
    set -a && source .env && set +a
    PYTHONUNBUFFERED=1 uv run --project daemon python scripts/rl_self_play.py \
        --num-iterations 5 --matches-per-iter 4

The starting checkpoint is the SFT v1 checkpoint by default. Override
with --sft-checkpoint if you have a different starting point.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from yomi_daemon.protocol import ActionDecision, DIVector, DecisionExtras
from yomi_daemon.rl.suite_eval import evaluate_suite
from yomi_daemon.rl.tinker_env import format_completion_from_decision
from yomi_daemon.rl.v3 import (
    LeagueConfig,
    ScheduledMatch,
    SuiteConfig,
    SuiteSeedBank,
    build_default_suite,
    build_head_to_head_config,
    build_suite_seed_bank,
    count_completed_runs,
    default_tinker_policy,
    load_champion_registry,
    load_league_config,
    load_suite_config,
    load_suite_seed_bank,
    schedule_matches,
    write_suite_config,
    write_suite_seed_bank,
)

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_BASE = REPO_ROOT / "runs" / "rl_self_play"
CONFIG_TEMPLATE = REPO_ROOT / "daemon" / "config" / "rl_self_play_template.json"
DEFAULT_EVAL_REGISTRY_PATH = REPO_ROOT / "runs" / "rl_v3" / "champion_registry.json"
DEFAULT_SFT_CHECKPOINT = (
    "tinker://d3d1239e-cb3f-5cf5-b675-3cbb2cd928c2:train:0/weights/yomi-sft-cowboy-v1"
)


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RolloutStep:
    """A single decision from one player in a completed match."""

    prompt_text: str
    action_json: str
    action_name: str
    player_id: str  # "p1" or "p2"
    turn_id: int
    my_hp: int
    opp_hp: int
    max_hp: int
    was_fallback: bool
    legal_actions_count: int


@dataclass(slots=True)
class RolloutMatch:
    match_id: str
    winner: str | None
    total_turns: int
    steps: list[RolloutStep] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ScoredStep:
    step: RolloutStep
    reward: float
    dense_reward: float
    terminal_reward: float
    return_to_go: float


@dataclass(frozen=True, slots=True)
class EvalArtifacts:
    suite: SuiteConfig
    seed_bank: SuiteSeedBank
    suite_path: Path
    seed_bank_path: Path


def build_action_decision(
    request_payload: dict,
    decision_payload: dict,
) -> ActionDecision:
    """Convert raw rollout payloads into an ActionDecision."""
    match_id = request_payload.get("match_id")
    turn_id = request_payload.get("turn_id")
    action = decision_payload.get("action")
    if not isinstance(match_id, str):
        raise ValueError("request_payload.match_id must be a string")
    if not isinstance(turn_id, int):
        raise ValueError("request_payload.turn_id must be an int")
    if not isinstance(action, str) or not action:
        raise ValueError("decision_payload.action must be a non-empty string")

    extra_raw = decision_payload.get("extra") or {}
    di_raw = extra_raw.get("di")
    di = (
        DIVector.from_dict(di_raw, context="decision_payload.extra.di")
        if di_raw
        else None
    )

    return ActionDecision(
        match_id=match_id,
        turn_id=turn_id,
        action=action,
        data=decision_payload.get("data"),
        extra=DecisionExtras(
            di=di,
            feint=bool(extra_raw.get("feint", False)),
            reverse=bool(extra_raw.get("reverse", False)),
            prediction=extra_raw.get("prediction"),
        ),
        reasoning=(
            decision_payload.get("reasoning")
            if isinstance(decision_payload.get("reasoning"), str)
            else None
        ),
        notes=(
            decision_payload.get("notes")
            if isinstance(decision_payload.get("notes"), str)
            else None
        ),
        fallback_reason=decision_payload.get("fallback_reason"),
    )


def action_json_from_payload(request_payload: dict, decision_payload: dict) -> str:
    """Build the canonical action-only completion for a rollout decision."""
    decision = build_action_decision(request_payload, decision_payload)
    return format_completion_from_decision(decision)


def read_completed_match(run_dir: Path) -> RolloutMatch | None:
    """Parse a completed match directory into a RolloutMatch.

    Returns None if the match didn't complete successfully.
    """
    result_file = run_dir / "result.json"
    decisions_file = run_dir / "decisions.jsonl"
    prompts_file = run_dir / "prompts.jsonl"
    if not all(f.exists() for f in (result_file, decisions_file, prompts_file)):
        return None

    result = json.loads(result_file.read_text())
    if result.get("status") != "completed":
        return None

    prompts: dict[tuple[str, int], str] = {}
    with open(prompts_file) as f:
        for line in f:
            prompt_record = json.loads(line)
            key = (prompt_record.get("player_id"), prompt_record.get("turn_id"))
            if isinstance(key[0], str) and isinstance(key[1], int):
                prompts[key] = prompt_record.get("prompt_text", "")

    steps: list[RolloutStep] = []
    with open(decisions_file) as f:
        for line in f:
            dec = json.loads(line)
            request_payload = dec.get("request_payload", {})
            decision_payload = dec.get("decision_payload", {})
            player_id = request_payload.get("player_id")
            turn_id = request_payload.get("turn_id")
            if not player_id or not isinstance(turn_id, int):
                continue

            fighters = request_payload.get("observation", {}).get("fighters", [])
            if len(fighters) != 2:
                continue
            p1_hp = int(fighters[0].get("hp", 0))
            p2_hp = int(fighters[1].get("hp", 0))
            p1_max_hp = int(fighters[0].get("max_hp", 750))
            p2_max_hp = int(fighters[1].get("max_hp", 750))
            my_hp = p1_hp if player_id == "p1" else p2_hp
            opp_hp = p2_hp if player_id == "p1" else p1_hp
            max_hp = p1_max_hp if player_id == "p1" else p2_max_hp

            action = decision_payload.get("action")
            if not isinstance(action, str) or not action:
                continue

            prompt_text = prompts.get((player_id, turn_id), "")
            if not prompt_text:
                continue

            legal_actions = request_payload.get("legal_actions")
            legal_actions_count = (
                len(legal_actions) if isinstance(legal_actions, list) else 0
            )

            try:
                action_json = action_json_from_payload(
                    request_payload, decision_payload
                )
            except Exception:
                continue

            steps.append(
                RolloutStep(
                    prompt_text=prompt_text,
                    action_json=action_json,
                    action_name=action,
                    player_id=player_id,
                    turn_id=turn_id,
                    my_hp=my_hp,
                    opp_hp=opp_hp,
                    max_hp=max_hp,
                    was_fallback=decision_payload.get("fallback_reason") is not None,
                    legal_actions_count=legal_actions_count,
                )
            )

    return RolloutMatch(
        match_id=result.get("match_id", run_dir.name),
        winner=result.get("winner"),
        total_turns=result.get("total_turns", len(steps)),
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


def is_meaningful_step(step: RolloutStep) -> bool:
    """Return True if a step is a real policy decision worth training on."""
    return (not step.was_fallback) and step.legal_actions_count > 1


def score_player_steps(
    steps: list[RolloutStep],
    *,
    player_id: str,
    winner: str | None,
    hp_delta_weight: float,
    terminal_reward: float,
    gamma: float,
) -> list[ScoredStep]:
    """Score one player's meaningful decisions within a match."""
    meaningful_steps = [step for step in steps if is_meaningful_step(step)]
    if not meaningful_steps:
        return []

    rewards: list[float] = []
    dense_rewards: list[float] = []
    terminal_rewards: list[float] = []

    previous_step: RolloutStep | None = None
    for step in meaningful_steps:
        dense = 0.0
        if previous_step is not None and hp_delta_weight > 0:
            damage_dealt = previous_step.opp_hp - step.opp_hp
            damage_taken = previous_step.my_hp - step.my_hp
            dense = (
                hp_delta_weight * (damage_dealt - damage_taken) / max(step.max_hp, 1)
            )
        rewards.append(dense)
        dense_rewards.append(dense)
        terminal_rewards.append(0.0)
        previous_step = step

    if winner == player_id:
        terminal_rewards[-1] = terminal_reward
        rewards[-1] += terminal_reward
    elif winner in {"p1", "p2"}:
        terminal_rewards[-1] = -terminal_reward
        rewards[-1] -= terminal_reward

    returns = [0.0] * len(meaningful_steps)
    running_return = 0.0
    for index in range(len(meaningful_steps) - 1, -1, -1):
        running_return = rewards[index] + gamma * running_return
        returns[index] = running_return

    return [
        ScoredStep(
            step=step,
            reward=reward,
            dense_reward=dense_reward,
            terminal_reward=terminal_component,
            return_to_go=return_to_go,
        )
        for step, reward, dense_reward, terminal_component, return_to_go in zip(
            meaningful_steps,
            rewards,
            dense_rewards,
            terminal_rewards,
            returns,
        )
    ]


def score_match_steps(
    match: RolloutMatch,
    *,
    hp_delta_weight: float,
    terminal_reward: float,
    gamma: float,
    train_player_ids: tuple[str, ...] = ("p1", "p2"),
) -> list[ScoredStep]:
    """Score all trainable players' meaningful decisions within a match."""
    scored_steps: list[ScoredStep] = []
    for player_id in train_player_ids:
        player_steps = [step for step in match.steps if step.player_id == player_id]
        scored_steps.extend(
            score_player_steps(
                player_steps,
                player_id=player_id,
                winner=match.winner,
                hp_delta_weight=hp_delta_weight,
                terminal_reward=terminal_reward,
                gamma=gamma,
            )
        )
    return scored_steps


def scale_advantages(
    returns: np.ndarray,
    *,
    scale_percentile: float,
    clip: float,
) -> tuple[np.ndarray, float]:
    """Center, robustly scale, and clip return-based advantages."""
    if returns.size == 0:
        return returns.astype(np.float32), 0.0

    centered = returns - returns.mean()
    scale = float(np.percentile(np.abs(centered), scale_percentile))
    if scale > 1e-6:
        normalized = centered / scale
    else:
        normalized = centered

    clipped = np.clip(normalized, -clip, clip)
    clip_fraction = float((~np.isclose(clipped, normalized)).mean())
    return clipped.astype(np.float32), clip_fraction


# ---------------------------------------------------------------------------
# Periodic suite evaluation
# ---------------------------------------------------------------------------


def should_run_periodic_eval(
    *,
    iteration_index: int,
    num_iterations: int,
    eval_every: int,
) -> bool:
    """Return True when an iteration should trigger a benchmark suite eval."""
    if eval_every <= 0:
        return False
    iter_number = iteration_index + 1
    return iter_number == num_iterations or (iter_number % eval_every == 0)


def prepare_eval_artifacts(
    args: argparse.Namespace,
    *,
    runs_root: Path,
) -> EvalArtifacts:
    """Load/build the benchmark suite and its fixed seed bank for this run."""
    eval_bundle_dir = runs_root / "benchmark_suite"
    eval_bundle_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_suite_path:
        suite = load_suite_config(Path(args.eval_suite_path))
    else:
        registry = load_champion_registry(Path(args.eval_registry_path))
        suite = build_default_suite(
            registry=registry,
            num_matches=args.eval_num_matches,
            prompt_version=args.learner_prompt_version,
            include_sft=not args.eval_no_include_sft,
            include_champion=not args.eval_no_include_champion,
            history_limit=args.eval_history_limit,
        )
    suite_path = eval_bundle_dir / "suite.json"
    write_suite_config(suite_path, suite)

    seed_bank_path = (
        Path(args.eval_seed_bank_path)
        if args.eval_seed_bank_path is not None
        else eval_bundle_dir / "seed_bank.json"
    )
    if seed_bank_path.exists():
        seed_bank = load_suite_seed_bank(seed_bank_path)
    else:
        seed_bank = build_suite_seed_bank(suite, seed_base=args.eval_seed_base)
        write_suite_seed_bank(seed_bank_path, seed_bank)

    return EvalArtifacts(
        suite=suite,
        seed_bank=seed_bank,
        suite_path=suite_path,
        seed_bank_path=seed_bank_path,
    )


# ---------------------------------------------------------------------------
# Match orchestration
# ---------------------------------------------------------------------------


def write_match_config(
    base_template: dict[str, Any], match_spec: ScheduledMatch
) -> Path:
    """Create a temp daemon config for a scheduled learner/opponent match."""
    template = build_head_to_head_config(
        base_template,
        p1_policy=match_spec.p1_policy,
        p2_policy=match_spec.p2_policy,
    )

    fd, path = tempfile.mkstemp(suffix=".json", prefix="rl_self_play_")
    import os as _os

    with _os.fdopen(fd, "w") as f:
        json.dump(template, f, indent=2)
    return Path(path)


def run_match_subprocess(
    config_path: Path,
    runs_dir: Path,
    trace_seed: int,
    timeout_s: int = 900,
    *,
    skip_mod_push: bool = True,
) -> bool:
    """Shell out to scripts/run_match.sh for one match. Returns True on success."""
    cmd = [
        str(REPO_ROOT / "scripts" / "run_match.sh"),
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
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, timeout=timeout_s, capture_output=False
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Match timed out after {timeout_s}s")
        return False


# ---------------------------------------------------------------------------
# Tokenization + advantage weighting
# ---------------------------------------------------------------------------


def build_training_datum(
    step: RolloutStep,
    *,
    renderer,
    max_length: int,
    advantage: float,
    train_on_what,
    conversation_to_datum,
):
    """Create a Tinker Datum with advantage-scaled loss weights."""
    conv = [
        {"role": "user", "content": step.prompt_text},
        {"role": "assistant", "content": step.action_json},
    ]
    datum = conversation_to_datum(
        cast(Any, conv),
        renderer,
        max_length=max_length,
        train_on_what=train_on_what,
    )

    import tinker

    orig_weights = datum.loss_fn_inputs["weights"]
    if hasattr(orig_weights, "to_numpy"):
        weights_np = orig_weights.to_numpy().astype(np.float32) * float(advantage)
    else:
        weights_np = np.array(
            orig_weights.tolist() if hasattr(orig_weights, "tolist") else orig_weights,
            dtype=np.float32,
        ) * float(advantage)
    new_weights = tinker.TensorData.from_numpy(weights_np)
    new_loss_inputs = dict(datum.loss_fn_inputs)
    new_loss_inputs["weights"] = new_weights
    return datum.model_copy(update={"loss_fn_inputs": new_loss_inputs})


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def train(args: argparse.Namespace) -> int:
    import tinker
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    print(f"Starting RL self-play from {args.sft_checkpoint}")
    print(f"Iterations: {args.num_iterations}, matches/iter: {args.matches_per_iter}")
    print(f"Learning rate: {args.lr}")
    print(f"HP delta weight: {args.hp_delta_weight}")
    print(f"Terminal reward: {args.terminal_reward} | gamma: {args.gamma}")
    print(f"Learner rollout prompt: {args.learner_prompt_version}")
    if args.eval_every > 0:
        print(
            f"Periodic suite eval: every {args.eval_every} iter(s), "
            f"{args.eval_num_matches} matches/entry"
        )
    print()

    service = tinker.ServiceClient()
    print("Creating training client from SFT checkpoint...")
    tc = await service.create_training_client_from_state_async(path=args.sft_checkpoint)
    tokenizer = tc.get_tokenizer()
    renderer = get_renderer("qwen3", tokenizer)
    print("Training client ready")
    print()

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    runs_root = RUNS_BASE / f"run_{run_id}"
    runs_root.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {runs_root}")
    print()

    sft_examples: list[tuple[str, str]] = []
    if args.sft_replay_ratio > 0 and args.sft_replay_data:
        sft_path = Path(args.sft_replay_data)
        if sft_path.exists():
            with open(sft_path) as f:
                for line in f:
                    obj = json.loads(line)
                    prompt = obj.get("prompt")
                    completion = obj.get("completion")
                    if isinstance(prompt, str) and isinstance(completion, str):
                        sft_examples.append((prompt, completion))
            print(f"Loaded {len(sft_examples)} SFT replay examples from {sft_path}")
            print()
        else:
            print(f"WARNING: SFT replay data not found at {sft_path}; disabling anchor")
            args.sft_replay_ratio = 0.0

    league: LeagueConfig | None = None
    if args.league_path:
        league_path = Path(args.league_path)
        league = load_league_config(league_path)
        print(
            f"Loaded league with {len(league.entries)} entries from {league_path}: "
            f"{', '.join(entry.entry_id for entry in league.entries)}"
        )
        print()

    base_template = json.loads(CONFIG_TEMPLATE.read_text())
    rng = np.random.default_rng(42)
    eval_artifacts: EvalArtifacts | None = None
    eval_index: list[dict[str, Any]] = []
    had_any_completed_match = False

    for iteration in range(args.num_iterations):
        iter_start = time.time()
        print(f"{'=' * 72}")
        print(f"  Iteration {iteration + 1}/{args.num_iterations}")
        print(f"{'=' * 72}")

        print("\n[1/5] Saving sampler weights...")
        save_future = await tc.save_weights_for_sampler_async(
            name=f"rl-selfplay-iter{iteration}"
        )
        save_result = await save_future.result_async()
        sampler_path = save_result.path
        print(f"  Sampler: {sampler_path}")

        print(f"\n[2/5] Running {args.matches_per_iter} scheduled matches...")
        iter_runs_dir = runs_root / f"iter_{iteration:03d}"
        iter_runs_dir.mkdir(parents=True, exist_ok=True)
        match_specs = schedule_matches(
            learner_model=sampler_path,
            matches_per_iter=args.matches_per_iter,
            iteration=iteration,
            rng=rng,
            league=league,
            learner_prompt_version=args.learner_prompt_version,
            learner_temperature=args.rollout_temperature,
            learner_max_tokens=args.rollout_max_tokens,
        )
        opponent_mix: dict[str, int] = {}
        for spec in match_specs:
            opponent_mix[spec.opponent_entry_id] = (
                opponent_mix.get(spec.opponent_entry_id, 0) + 1
            )
        print(
            "  Opponent mix: "
            + ", ".join(
                f"{entry_id}={count}"
                for entry_id, count in sorted(opponent_mix.items())
            )
        )

        completed = 0
        failed = 0
        for spec in match_specs:
            schedule_dir = iter_runs_dir / f"match_{spec.match_index:03d}"
            schedule_dir.mkdir(parents=True, exist_ok=True)
            (schedule_dir / "match_spec.json").write_text(
                json.dumps(spec.to_dict(), indent=2) + "\n"
            )
            completed_before = count_completed_runs(schedule_dir)
            t0 = time.time()
            match_completed = False
            for attempt_index in range(args.match_max_attempts):
                config_path = write_match_config(base_template, spec)
                ok = run_match_subprocess(
                    config_path,
                    schedule_dir,
                    spec.trace_seed,
                    timeout_s=args.match_timeout_s,
                    skip_mod_push=args.skip_mod_push,
                )
                config_path.unlink(missing_ok=True)
                completed_after = count_completed_runs(schedule_dir)
                if ok and completed_after > completed_before:
                    match_completed = True
                    break
                if args.match_max_attempts > 1:
                    print(
                        f"    attempt {attempt_index + 1}/{args.match_max_attempts} did not complete"
                    )
            elapsed = time.time() - t0
            side_desc = (
                "both-sides"
                if len(spec.train_player_ids) == 2
                else spec.learner_player_id
            )
            if match_completed:
                completed += 1
                print(
                    f"  Match {spec.match_index + 1}/{args.matches_per_iter} "
                    f"vs {spec.opponent_entry_id} train={side_desc} OK ({elapsed:.0f}s)"
                )
            else:
                failed += 1
                print(
                    f"  Match {spec.match_index + 1}/{args.matches_per_iter} "
                    f"vs {spec.opponent_entry_id} train={side_desc} FAILED ({elapsed:.0f}s)"
                )

        if completed == 0:
            print(
                f"  No successful matches; skipping training for iter {iteration + 1}"
            )
            continue

        had_any_completed_match = True

        print("\n[3/5] Parsing trajectories...")
        parsed_matches: list[tuple[RolloutMatch, ScheduledMatch]] = []
        for schedule_dir in sorted(iter_runs_dir.iterdir()):
            if not schedule_dir.is_dir():
                continue
            spec_file = schedule_dir / "match_spec.json"
            if not spec_file.exists():
                continue
            spec = ScheduledMatch.from_dict(json.loads(spec_file.read_text()))
            completed_run: RolloutMatch | None = None
            for run_dir in sorted(
                path for path in schedule_dir.iterdir() if path.is_dir()
            ):
                match = read_completed_match(run_dir)
                if match is not None:
                    completed_run = match
                    break
            if completed_run is not None:
                parsed_matches.append((completed_run, spec))
        print(f"  Parsed {len(parsed_matches)} matches")

        matches = [match for match, _ in parsed_matches]
        p1_wins = sum(1 for match in matches if match.winner == "p1")
        p2_wins = sum(1 for match in matches if match.winner == "p2")
        draws = len(matches) - p1_wins - p2_wins
        total_steps = sum(len(match.steps) for match in matches)
        fallback_steps = sum(
            sum(1 for step in match.steps if step.was_fallback) for match in matches
        )
        fallback_rate = fallback_steps / max(total_steps, 1)
        print(
            f"  P1: {p1_wins}W, P2: {p2_wins}W, draws: {draws} | "
            f"fallback rate: {fallback_rate:.1%}"
        )
        if fallback_rate > 0.20:
            print(
                f"  ABORT: fallback rate {fallback_rate:.1%} > 20% — model is degrading. "
                "Stopping before more damage."
            )
            break

        non_fallback_steps = [
            step
            for match, spec in parsed_matches
            for step in match.steps
            if not step.was_fallback and step.player_id in spec.train_player_ids
        ]
        meaningful_steps = [
            step for step in non_fallback_steps if is_meaningful_step(step)
        ]
        filtered_steps = len(non_fallback_steps) - len(meaningful_steps)

        scored_steps: list[ScoredStep] = []
        for match, spec in parsed_matches:
            scored_steps.extend(
                score_match_steps(
                    match,
                    hp_delta_weight=args.hp_delta_weight,
                    terminal_reward=args.terminal_reward,
                    gamma=args.gamma,
                    train_player_ids=spec.train_player_ids,
                )
            )

        if not scored_steps:
            print("  No meaningful training steps; skipping iteration")
            continue

        rewards_arr = np.array(
            [scored.reward for scored in scored_steps], dtype=np.float32
        )
        dense_arr = np.array(
            [scored.dense_reward for scored in scored_steps], dtype=np.float32
        )
        terminal_arr = np.array(
            [scored.terminal_reward for scored in scored_steps], dtype=np.float32
        )
        returns_arr = np.array(
            [scored.return_to_go for scored in scored_steps], dtype=np.float32
        )
        advantages, clip_fraction = scale_advantages(
            returns_arr,
            scale_percentile=args.adv_scale_percentile,
            clip=args.adv_clip,
        )

        reward_var = float(rewards_arr.var())
        dense_var_fraction = (
            float(dense_arr.var() / reward_var) if reward_var > 1e-8 else 0.0
        )
        terminal_var_fraction = (
            float(terminal_arr.var() / reward_var) if reward_var > 1e-8 else 0.0
        )

        print(
            f"  Non-fallback steps: {len(non_fallback_steps)} | meaningful: {len(scored_steps)} "
            f"({len(scored_steps) / max(len(non_fallback_steps), 1):.1%}) | "
            f"filtered trivial: {filtered_steps}"
        )
        print(
            f"  reward stats: min={rewards_arr.min():.3f} max={rewards_arr.max():.3f} "
            f"mean={rewards_arr.mean():.3f} std={rewards_arr.std():.3f}"
        )
        print(
            f"  return stats: min={returns_arr.min():.3f} max={returns_arr.max():.3f} "
            f"mean={returns_arr.mean():.3f} std={returns_arr.std():.3f}"
        )
        print(
            f"  advantage stats: min={advantages.min():.2f} max={advantages.max():.2f} | "
            f"clip_fraction={clip_fraction:.1%}"
        )
        print(
            f"  dense reward: mean={dense_arr.mean():.3f} std={dense_arr.std():.3f} "
            f"var_frac={dense_var_fraction:.2f} | terminal reward: mean={terminal_arr.mean():.3f} "
            f"std={terminal_arr.std():.3f} var_frac={terminal_var_fraction:.2f}"
        )
        if float(np.abs(advantages).max()) < 1e-6:
            print("  All advantages zero; skipping training for this iter")
            continue

        print("\n[4/5] Tokenizing with advantage-scaled weights...")
        training_data = []
        skipped = 0
        for scored_step, advantage in zip(scored_steps, advantages):
            try:
                datum = build_training_datum(
                    scored_step.step,
                    renderer=renderer,
                    max_length=args.max_length,
                    advantage=float(advantage),
                    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                    conversation_to_datum=conversation_to_datum,
                )
                training_data.append(datum)
            except Exception as exc:
                skipped += 1
                if skipped <= 3:
                    print(f"  Skip: {exc}")
        print(f"  Built {len(training_data)} training data items ({skipped} skipped)")

        if not training_data:
            print("  No training data after tokenization; skipping iteration")
            continue

        print("\n[5/5] Training...")
        batch_size = args.batch_size
        num_batches = (len(training_data) + batch_size - 1) // batch_size
        losses: list[float] = []
        for batch_index in range(num_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, len(training_data))
            batch = training_data[start:end]

            fwd = await tc.forward_backward_async(batch, "cross_entropy")
            opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=args.lr))
            result = await fwd.result_async()
            await opt.result_async()

            logprobs = np.concatenate(
                [out["logprobs"].tolist() for out in result.loss_fn_outputs]
            )
            weights = np.concatenate(
                [
                    datum.loss_fn_inputs["weights"].tolist()
                    if hasattr(datum.loss_fn_inputs["weights"], "tolist")
                    else datum.loss_fn_inputs["weights"]
                    for datum in batch
                ]
            )
            denom = max(np.abs(weights).sum(), 1e-8)
            loss = -float(np.dot(logprobs, weights)) / denom
            losses.append(loss)
            if (batch_index + 1) % 10 == 0 or batch_index == 0:
                print(f"  batch {batch_index + 1}/{num_batches} loss={loss:.4f}")

        if args.sft_replay_ratio > 0 and sft_examples:
            num_sft_batches = max(1, int(round(num_batches * args.sft_replay_ratio)))
            n_needed = num_sft_batches * batch_size
            replace = len(sft_examples) < n_needed
            indices = rng.choice(len(sft_examples), size=n_needed, replace=replace)
            sft_data = []
            for index in indices:
                prompt, completion = sft_examples[int(index)]
                conv = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
                try:
                    datum = conversation_to_datum(
                        cast(Any, conv),
                        renderer,
                        max_length=args.max_length,
                        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                    )
                    sft_data.append(datum)
                except Exception:
                    pass
            print(
                f"\n[5b] SFT replay anchor: {len(sft_data)} examples "
                f"({num_sft_batches} batches)"
            )
            sft_losses: list[float] = []
            for batch_start in range(0, len(sft_data), batch_size):
                batch = sft_data[batch_start : batch_start + batch_size]
                if not batch:
                    continue
                fwd = await tc.forward_backward_async(batch, "cross_entropy")
                opt = await tc.optim_step_async(
                    tinker.AdamParams(learning_rate=args.lr)
                )
                result = await fwd.result_async()
                await opt.result_async()
                logprobs = np.concatenate(
                    [out["logprobs"].tolist() for out in result.loss_fn_outputs]
                )
                weights = np.concatenate(
                    [
                        datum.loss_fn_inputs["weights"].tolist()
                        if hasattr(datum.loss_fn_inputs["weights"], "tolist")
                        else datum.loss_fn_inputs["weights"]
                        for datum in batch
                    ]
                )
                denom = max(np.abs(weights).sum(), 1e-8)
                sft_losses.append(-float(np.dot(logprobs, weights)) / denom)
            if sft_losses:
                print(
                    f"  SFT replay avg_loss={float(np.mean(sft_losses)):.4f} "
                    f"(unit weights; lower = higher log-prob on SFT actions)"
                )

        persistent_checkpoint_path: str | None = None
        if (
            iteration + 1
        ) % args.save_every == 0 or iteration + 1 == args.num_iterations:
            print(f"Saving persistent checkpoint (iter {iteration + 1})...")
            checkpoint_future = await tc.save_state_async(
                name=f"rl-selfplay-iter{iteration + 1}",
                ttl_seconds=30 * 24 * 3600,
            )
            checkpoint_result = await checkpoint_future.result_async()
            persistent_checkpoint_path = checkpoint_result.path
            print(f"  Saved: {checkpoint_result.path}")
            print()

        eval_sampler_path: str | None = None
        eval_result_path: str | None = None
        eval_score: float | None = None
        if should_run_periodic_eval(
            iteration_index=iteration,
            num_iterations=args.num_iterations,
            eval_every=args.eval_every,
        ):
            if eval_artifacts is None:
                eval_artifacts = prepare_eval_artifacts(args, runs_root=runs_root)
                print(
                    f"Prepared benchmark suite at {eval_artifacts.suite_path} "
                    f"with seed bank {eval_artifacts.seed_bank_path}"
                )
                print()

            print(f"[6/6] Benchmark suite eval (iter {iteration + 1})...")
            eval_sampler_future = await tc.save_weights_for_sampler_async(
                name=f"rl-selfplay-iter{iteration + 1}-eval"
            )
            eval_sampler_result = await eval_sampler_future.result_async()
            eval_sampler_path = eval_sampler_result.path
            eval_policy = default_tinker_policy(
                policy_id=f"rl/selfplay-eval-{iteration + 1}",
                model=eval_sampler_path,
                prompt_version=args.learner_prompt_version,
                temperature=args.rollout_temperature,
                max_tokens=args.rollout_max_tokens,
                label=f"rl-selfplay-iter{iteration + 1}",
            )
            eval_output_dir = runs_root / "evals" / f"iter_{iteration + 1:03d}"
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            eval_result = evaluate_suite(
                repo_root=REPO_ROOT,
                candidate_policy=eval_policy,
                suite=eval_artifacts.suite,
                output_root=eval_output_dir,
                base_template_path=CONFIG_TEMPLATE,
                match_timeout_s=args.match_timeout_s,
                seed_base=args.eval_seed_base,
                seed_bank=eval_artifacts.seed_bank,
                match_max_attempts=args.match_max_attempts,
            )
            eval_result_path_obj = eval_output_dir / "eval_results.json"
            eval_result_path_obj.write_text(json.dumps(eval_result, indent=2) + "\n")
            eval_result_path = str(eval_result_path_obj)
            eval_score = float(eval_result["aggregate_score"])
            eval_index.append(
                {
                    "iteration": iteration + 1,
                    "candidate_model": eval_sampler_path,
                    "aggregate_score": eval_score,
                    "result_path": eval_result_path,
                }
            )
            eval_index_path = runs_root / "evals" / "index.json"
            eval_index_path.parent.mkdir(parents=True, exist_ok=True)
            eval_index_path.write_text(json.dumps(eval_index, indent=2) + "\n")
            best_eval = max(eval_index, key=lambda item: float(item["aggregate_score"]))
            (runs_root / "best_eval.json").write_text(
                json.dumps(best_eval, indent=2) + "\n"
            )
            print(f"  Eval aggregate score={eval_score:.3f}")
            print(f"  Eval results: {eval_result_path}")
            print()

        iter_time = time.time() - iter_start
        avg_loss = float(np.mean(losses)) if losses else 0.0
        iteration_summary = {
            "iteration": iteration + 1,
            "pretrain_sampler_path": sampler_path,
            "persistent_checkpoint_path": persistent_checkpoint_path,
            "eval_sampler_path": eval_sampler_path,
            "eval_result_path": eval_result_path,
            "eval_score": eval_score,
            "completed_matches": completed,
            "failed_matches": failed,
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "fallback_rate": fallback_rate,
            "opponent_mix": opponent_mix,
            "non_fallback_steps": len(non_fallback_steps),
            "meaningful_steps": len(scored_steps),
            "filtered_trivial_steps": filtered_steps,
            "clip_fraction": clip_fraction,
            "reward_stats": {
                "min": float(rewards_arr.min()),
                "max": float(rewards_arr.max()),
                "mean": float(rewards_arr.mean()),
                "std": float(rewards_arr.std()),
            },
            "return_stats": {
                "min": float(returns_arr.min()),
                "max": float(returns_arr.max()),
                "mean": float(returns_arr.mean()),
                "std": float(returns_arr.std()),
            },
            "advantage_stats": {
                "min": float(advantages.min()),
                "max": float(advantages.max()),
            },
            "dense_reward_variance_fraction": dense_var_fraction,
            "terminal_reward_variance_fraction": terminal_var_fraction,
            "training_examples": len(training_data),
            "avg_loss": avg_loss,
            "iteration_time_s": iter_time,
        }
        (iter_runs_dir / "iteration_summary.json").write_text(
            json.dumps(iteration_summary, indent=2) + "\n"
        )

        print()
        print(
            f"Iteration {iteration + 1} done: "
            f"matches={completed} ({p1_wins}-{p2_wins}) "
            f"steps={len(training_data)} "
            f"avg_loss={avg_loss:.4f} "
            f"time={iter_time:.0f}s"
        )
        if eval_score is not None:
            print(f"  eval_score={eval_score:.3f}")
        print()

    print("RL self-play complete.")
    if not had_any_completed_match:
        print("ERROR: run finished without any completed matches.")
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="RL self-play training on Tinker")
    parser.add_argument(
        "--sft-checkpoint",
        default=DEFAULT_SFT_CHECKPOINT,
        help="Tinker checkpoint path to start from",
    )
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--matches-per-iter", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--league-path",
        default=None,
        help="Optional path to an RL v3 league JSON file. If omitted, uses mirror only.",
    )
    parser.add_argument(
        "--learner-prompt-version",
        default="strategic_rl_v1",
        help="Prompt version used by learner rollouts",
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for learner rollouts",
    )
    parser.add_argument(
        "--rollout-max-tokens",
        type=int,
        default=256,
        help="Max tokens for learner rollouts",
    )
    parser.add_argument(
        "--adv-clip",
        type=float,
        default=1.0,
        help="Clip robustly-scaled advantages to +/- this value",
    )
    parser.add_argument(
        "--adv-scale-percentile",
        type=float,
        default=90.0,
        help="Percentile used to robustly scale absolute centered returns",
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--hp-delta-weight",
        type=float,
        default=0.05,
        help="Dense reward shaping weight for HP delta",
    )
    parser.add_argument(
        "--terminal-reward",
        type=float,
        default=1.0,
        help="Terminal win/loss reward added only to the last meaningful step",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for return-to-go",
    )
    parser.add_argument(
        "--match-timeout-s",
        type=int,
        default=900,
        help="Timeout in seconds for each self-play match subprocess",
    )
    parser.add_argument(
        "--match-max-attempts",
        type=int,
        default=1,
        help="Retry each scheduled match up to this many times until a completed run is recorded",
    )
    parser.add_argument(
        "--skip-mod-push",
        action="store_true",
        help="Reuse the currently installed VM mod instead of pushing the local mod before each match",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help=(
            "Run benchmark suite evaluation every N iterations and on the final iteration. "
            "0 disables periodic evaluation."
        ),
    )
    parser.add_argument(
        "--eval-suite-path",
        default=None,
        help="Optional explicit suite JSON for periodic evaluation",
    )
    parser.add_argument(
        "--eval-registry-path",
        default=str(DEFAULT_EVAL_REGISTRY_PATH),
        help="Champion registry used when building the default periodic eval suite",
    )
    parser.add_argument(
        "--eval-num-matches",
        type=int,
        default=6,
        help="Matches per suite entry for periodic evaluation when auto-building the suite",
    )
    parser.add_argument(
        "--eval-history-limit",
        type=int,
        default=3,
        help="Historical champions to include when auto-building the periodic eval suite",
    )
    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=4000,
        help="Base seed used to generate the periodic eval seed bank",
    )
    parser.add_argument(
        "--eval-seed-bank-path",
        default=None,
        help="Optional seed bank path for periodic evaluation; reused if it already exists",
    )
    parser.add_argument(
        "--eval-no-include-sft",
        action="store_true",
        help="Exclude frozen SFT from the auto-built periodic eval suite",
    )
    parser.add_argument(
        "--eval-no-include-champion",
        action="store_true",
        help="Exclude current champion from the auto-built periodic eval suite",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save a persistent checkpoint every N iterations",
    )
    parser.add_argument(
        "--sft-replay-data",
        default="runs/rl_sft_cowboy/sft_training_data_action_only.jsonl",
        help="Path to SFT training jsonl with 'prompt'/'completion' fields",
    )
    parser.add_argument(
        "--sft-replay-ratio",
        type=float,
        default=0.5,
        help=(
            "Ratio of SFT replay batches per RL batch per iter (0 = off). "
            "Acts as a KL-style anchor keeping the policy near SFT v1."
        ),
    )
    args = parser.parse_args()

    raise SystemExit(asyncio.run(train(args)))


if __name__ == "__main__":
    main()
