"""RL self-play training on Tinker.

Starts from the SFT v1 checkpoint and improves via self-play + policy
gradient with a baseline (REINFORCE-style). Each iteration:

  1. Save current weights as a Tinker sampler checkpoint
  2. Run N self-play matches (both P1 and P2 use the current policy)
  3. Parse trajectories from runs/, compute rewards (win/loss + HP shaping)
  4. Compute advantages with a group baseline (reward - batch mean)
  5. Tokenize with advantage-scaled weights
  6. Forward/backward with cross_entropy (policy gradient)
  7. Evaluate against frozen SFT v1 every N iterations

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

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "runs" / "rl_self_play"
CONFIG_TEMPLATE = REPO_ROOT / "daemon" / "config" / "rl_self_play_template.json"
TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
DEFAULT_SFT_CHECKPOINT = (
    "tinker://d3d1239e-cb3f-5cf5-b675-3cbb2cd928c2:train:0/weights/yomi-sft-cowboy-v1"
)


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------


@dataclass
class RolloutStep:
    """A single decision from one player in a completed match."""

    prompt_text: str
    action_json: str  # the raw completion JSON
    player_id: str  # "p1" or "p2"
    turn_id: int
    my_hp: int
    opp_hp: int
    was_fallback: bool


@dataclass
class RolloutMatch:
    match_id: str
    winner: str | None
    total_turns: int
    steps: list[RolloutStep] = field(default_factory=list)


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

    # Index prompts by (player_id, turn_id)
    prompts: dict[tuple[str, int], str] = {}
    with open(prompts_file) as f:
        for line in f:
            p = json.loads(line)
            key = (p.get("player_id"), p.get("turn_id"))
            if isinstance(key[0], str) and isinstance(key[1], int):
                prompts[key] = p.get("prompt_text", "")

    steps: list[RolloutStep] = []
    with open(decisions_file) as f:
        for line in f:
            dec = json.loads(line)
            rp = dec.get("request_payload", {})
            dp = dec.get("decision_payload", {})
            player_id = rp.get("player_id")
            turn_id = rp.get("turn_id")
            if not player_id or not isinstance(turn_id, int):
                continue

            fighters = rp.get("observation", {}).get("fighters", [])
            if len(fighters) != 2:
                continue
            p1_hp = int(fighters[0].get("hp", 0))
            p2_hp = int(fighters[1].get("hp", 0))
            my_hp = p1_hp if player_id == "p1" else p2_hp
            opp_hp = p2_hp if player_id == "p1" else p1_hp

            action = dp.get("action")
            if not action:
                continue

            # Reconstruct the completion JSON the way format_completion_from_decision
            # would have emitted it during training.
            completion_obj: dict = {"action": action}
            if dp.get("data") is not None:
                completion_obj["data"] = dp["data"]
            extra = dp.get("extra") or {}
            extra_obj: dict = {}
            if extra.get("di") is not None:
                extra_obj["di"] = extra["di"]
            if extra.get("feint"):
                extra_obj["feint"] = True
            if extra.get("reverse"):
                extra_obj["reverse"] = True
            if extra_obj:
                completion_obj["extra"] = extra_obj
            if dp.get("reasoning"):
                completion_obj["reasoning"] = dp["reasoning"]

            prompt_text = prompts.get((player_id, turn_id), "")
            if not prompt_text:
                continue

            steps.append(
                RolloutStep(
                    prompt_text=prompt_text,
                    action_json=json.dumps(completion_obj, separators=(",", ":")),
                    player_id=player_id,
                    turn_id=turn_id,
                    my_hp=my_hp,
                    opp_hp=opp_hp,
                    was_fallback=dp.get("fallback_reason") is not None,
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


def compute_step_rewards(
    match: RolloutMatch,
    *,
    win_reward: float = 1.0,
    loss_reward: float = -1.0,
    hp_delta_weight: float = 0.0,
    max_hp: int = 750,
) -> list[float]:
    """Per-step reward for each step in the match.

    Terminal: apply win_reward to all steps from the winning side and
    loss_reward to all steps from the losing side. (Every step in a
    winning match gets credited; this is high bias but low variance
    compared to only rewarding the final step.)

    Optional dense: add hp_delta_weight * (opp_hp_delta - my_hp_delta) /
    max_hp per step, using the HP trajectory from this player's
    perspective. Off by default.
    """
    rewards = [0.0] * len(match.steps)

    # Terminal: credit every step from the winning side
    if match.winner:
        for i, step in enumerate(match.steps):
            if step.player_id == match.winner:
                rewards[i] += win_reward
            else:
                rewards[i] += loss_reward

    if hp_delta_weight > 0:
        # Compute HP deltas from the perspective of each player across their
        # consecutive decisions. We approximate by comparing my_hp and opp_hp
        # between consecutive steps for the same player.
        prev_by_player: dict[str, tuple[int, int]] = {}
        for i, step in enumerate(match.steps):
            prev = prev_by_player.get(step.player_id)
            if prev is not None:
                prev_my, prev_opp = prev
                my_delta = step.my_hp - prev_my  # negative = took damage
                opp_delta = step.opp_hp - prev_opp
                # Want negative opp_delta (damage dealt) to be positive reward
                rewards[i] += hp_delta_weight * (-(opp_delta) - (-(my_delta))) / max_hp
            prev_by_player[step.player_id] = (step.my_hp, step.opp_hp)

    return rewards


# ---------------------------------------------------------------------------
# Match orchestration
# ---------------------------------------------------------------------------


def write_self_play_config(sampler_path: str) -> Path:
    """Create a temp daemon config with both policies using the given sampler."""
    template = json.loads(CONFIG_TEMPLATE.read_text())
    for policy_id in ("rl/self-play-p1", "rl/self-play-p2"):
        template["policies"][policy_id]["model"] = sampler_path
        template["policies"][policy_id]["options"]["base_url"] = TINKER_OAI_BASE_URL

    fd, path = tempfile.mkstemp(suffix=".json", prefix="rl_self_play_")
    import os as _os

    with _os.fdopen(fd, "w") as f:
        json.dump(template, f, indent=2)
    return Path(path)


def run_match_subprocess(
    config_path: Path, runs_dir: Path, trace_seed: int, timeout_s: int = 900
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
        "--skip-mod-push",
    ]
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
        conv,
        renderer,
        max_length=max_length,
        train_on_what=train_on_what,
    )
    # Scale the per-token weights by the advantage so that:
    #   loss = -sum(weight * logprob)
    # becomes
    #   loss = -sum(advantage * weight * logprob)
    # which is the REINFORCE objective (maximize advantage * log p).
    import tinker

    orig_weights = datum.loss_fn_inputs["weights"]
    # Convert to numpy, scale by advantage, wrap back in TensorData.
    if hasattr(orig_weights, "to_numpy"):
        w_np = orig_weights.to_numpy().astype(np.float32) * float(advantage)
    else:
        w_np = np.array(
            orig_weights.tolist() if hasattr(orig_weights, "tolist") else orig_weights,
            dtype=np.float32,
        ) * float(advantage)
    new_weights = tinker.TensorData.from_numpy(w_np)
    # Datum is a frozen pydantic model; rebuild loss_fn_inputs via model_copy.
    new_loss_inputs = dict(datum.loss_fn_inputs)
    new_loss_inputs["weights"] = new_weights
    return datum.model_copy(update={"loss_fn_inputs": new_loss_inputs})


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def train(args: argparse.Namespace) -> None:
    import tinker
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    print(f"Starting RL self-play from {args.sft_checkpoint}")
    print(f"Iterations: {args.num_iterations}, matches/iter: {args.matches_per_iter}")
    print(f"Learning rate: {args.lr}")
    print()

    service = tinker.ServiceClient()
    print("Creating training client from SFT checkpoint...")
    tc = await service.create_training_client_from_state_async(
        path=args.sft_checkpoint,
    )
    tokenizer = tc.get_tokenizer()
    renderer = get_renderer("qwen3", tokenizer)
    print("Training client ready")
    print()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    for iteration in range(args.num_iterations):
        iter_start = time.time()
        print(f"{'=' * 72}")
        print(f"  Iteration {iteration + 1}/{args.num_iterations}")
        print(f"{'=' * 72}")

        # 1. Save current weights as a sampler checkpoint
        print("\n[1/5] Saving sampler weights...")
        save_future = await tc.save_weights_for_sampler_async(
            name=f"rl-selfplay-iter{iteration}"
        )
        save_result = await save_future.result_async()
        sampler_path = save_result.path
        print(f"  Sampler: {sampler_path}")

        # 2. Run self-play matches
        print(f"\n[2/5] Running {args.matches_per_iter} self-play matches...")
        config_path = write_self_play_config(sampler_path)
        iter_runs_dir = RUNS_ROOT / f"iter_{iteration:03d}"
        iter_runs_dir.mkdir(parents=True, exist_ok=True)

        completed, failed = 0, 0
        for mi in range(args.matches_per_iter):
            seed = 20000 + iteration * 1000 + mi
            t0 = time.time()
            ok = run_match_subprocess(config_path, iter_runs_dir, seed)
            elapsed = time.time() - t0
            if ok:
                completed += 1
                print(f"  Match {mi + 1}/{args.matches_per_iter} OK ({elapsed:.0f}s)")
            else:
                failed += 1
                print(
                    f"  Match {mi + 1}/{args.matches_per_iter} FAILED ({elapsed:.0f}s)"
                )
        config_path.unlink(missing_ok=True)

        if completed == 0:
            print(f"  No successful matches; skipping training for iter {iteration + 1}")
            continue

        # 3. Parse trajectories and compute rewards
        print("\n[3/5] Parsing trajectories...")
        matches: list[RolloutMatch] = []
        for d in sorted(iter_runs_dir.iterdir()):
            if not d.is_dir():
                continue
            m = read_completed_match(d)
            if m is not None:
                matches.append(m)
        print(f"  Parsed {len(matches)} matches")

        # Summary
        p1_wins = sum(1 for m in matches if m.winner == "p1")
        p2_wins = sum(1 for m in matches if m.winner == "p2")
        draws = len(matches) - p1_wins - p2_wins
        print(f"  P1: {p1_wins}W, P2: {p2_wins}W, draws: {draws}")

        # Per-step rewards
        all_steps: list[tuple[RolloutStep, float]] = []
        for match in matches:
            rewards = compute_step_rewards(
                match,
                hp_delta_weight=args.hp_delta_weight,
            )
            for step, r in zip(match.steps, rewards):
                if step.was_fallback:
                    continue  # don't train on fallback decisions
                all_steps.append((step, r))

        if not all_steps:
            print("  No valid training steps; skipping iteration")
            continue

        rewards_arr = np.array([r for _, r in all_steps], dtype=np.float32)
        baseline = float(rewards_arr.mean())
        advantages = rewards_arr - baseline
        std = float(advantages.std())
        if std > 1e-8:
            advantages = advantages / std
        print(
            f"  Steps: {len(all_steps)} | mean_r: {rewards_arr.mean():.3f} | "
            f"std: {rewards_arr.std():.3f} | adv: "
            f"[{advantages.min():.2f}, {advantages.max():.2f}]"
        )

        # 4. Tokenize
        print("\n[4/5] Tokenizing with advantage-scaled weights...")
        training_data = []
        skipped = 0
        for (step, _), advantage in zip(all_steps, advantages):
            try:
                datum = build_training_datum(
                    step,
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

        # 5. Train
        print("\n[5/5] Training...")
        batch_size = args.batch_size
        num_batches = (len(training_data) + batch_size - 1) // batch_size
        losses = []
        for bi in range(num_batches):
            start = bi * batch_size
            end = min(start + batch_size, len(training_data))
            batch = training_data[start:end]

            fwd = await tc.forward_backward_async(batch, "cross_entropy")
            opt = await tc.optim_step_async(
                tinker.AdamParams(learning_rate=args.lr)
            )
            result = await fwd.result_async()
            await opt.result_async()

            # Compute loss for logging
            logprobs = np.concatenate(
                [out["logprobs"].tolist() for out in result.loss_fn_outputs]
            )
            weights = np.concatenate(
                [
                    d.loss_fn_inputs["weights"].tolist()
                    if hasattr(d.loss_fn_inputs["weights"], "tolist")
                    else d.loss_fn_inputs["weights"]
                    for d in batch
                ]
            )
            # For policy gradient, the scalar loss is -sum(advantage * logprob * mask)
            # Use abs sum of weights as denominator to avoid sign issues
            denom = max(np.abs(weights).sum(), 1e-8)
            loss = -float(np.dot(logprobs, weights)) / denom
            losses.append(loss)
            if (bi + 1) % 10 == 0 or bi == 0:
                print(f"  batch {bi + 1}/{num_batches} loss={loss:.4f}")

        iter_time = time.time() - iter_start
        avg_loss = float(np.mean(losses)) if losses else 0.0
        print()
        print(
            f"Iteration {iteration + 1} done: "
            f"matches={completed} ({p1_wins}-{p2_wins}) "
            f"steps={len(training_data)} "
            f"avg_loss={avg_loss:.4f} "
            f"time={iter_time:.0f}s"
        )
        print()

        # Save persistent checkpoint every `save_every` iterations
        if (iteration + 1) % args.save_every == 0 or iteration + 1 == args.num_iterations:
            print(f"Saving persistent checkpoint (iter {iteration + 1})...")
            cp_future = await tc.save_state_async(
                name=f"rl-selfplay-iter{iteration + 1}",
                ttl_seconds=30 * 24 * 3600,
            )
            cp_result = await cp_future.result_async()
            print(f"  Saved: {cp_result.path}")
            print()

    print("RL self-play complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL self-play training on Tinker")
    parser.add_argument(
        "--sft-checkpoint",
        default=DEFAULT_SFT_CHECKPOINT,
        help="Tinker checkpoint path to start from",
    )
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--matches-per-iter", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--hp-delta-weight",
        type=float,
        default=0.0,
        help="Dense reward shaping weight for HP delta (0 = sparse only)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save a persistent checkpoint every N iterations",
    )
    args = parser.parse_args()

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
