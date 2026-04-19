#!/usr/bin/env python3
"""Run a replay-recorded best-of-7 showmatch between the current champion and Gemini.

This script uses the official current champion from runs/rl_v3/champion_registry.json,
plays a best-of-7 against Gemini with alternating sides, and requires replay.mp4 to
exist for every completed game so the results are ready for editing/social posting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from yomi_daemon.rl.v3 import (
    PolicySpec,
    build_head_to_head_config,
    load_champion_registry,
    load_json_document,
    timestamp_utc,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "runs" / "rl_v3" / "champion_registry.json"
DEFAULT_BASE_TEMPLATE = REPO_ROOT / "daemon" / "config" / "rl_self_play_template.json"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "runs" / "showmatches"


def _gemini_policy(*, policy_id: str, model: str, label: str) -> PolicySpec:
    return PolicySpec(
        policy_id=policy_id,
        provider="openrouter",
        model=model,
        prompt_version="strategic_v1",
        credential_env_var="OPENROUTER_API_KEY",
        temperature=0.7,
        max_tokens=4096,
        options={
            "title": "budok-ai",
            "categories": "game",
            "reasoning_effort": "low",
            "response_parser": {
                "enable_correction_retry": True,
                "max_correction_retries": 1,
            },
        },
        label=label,
    )


def _clone_policy(policy: PolicySpec, *, policy_id: str, label: str | None = None) -> PolicySpec:
    return replace(policy, policy_id=policy_id, label=label or policy.label)


def _latest_completed_or_failed_result(game_root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(game_root.rglob("result.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = json.loads(path.read_text())
        if payload.get("status") in {"completed", "failed"}:
            return path, payload
    raise RuntimeError(f"No completed/failed result.json found under {game_root}")


def _candidate_outcome(result: dict[str, Any], *, candidate_side: str) -> str:
    winner = result.get("winner")
    if winner == candidate_side:
        return "win"
    if winner in {"p1", "p2"}:
        return "loss"
    return "draw"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a replay-recorded best-of-7 champion vs Gemini showmatch")
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--base-template", type=Path, default=DEFAULT_BASE_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gemini-model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--match-timeout-s", type=int, default=300)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--skip-mod-push",
        action="store_true",
        help="Reuse the installed VM mod instead of pushing the local mod before each game",
    )
    args = parser.parse_args()

    registry = load_champion_registry(args.registry_path)
    if registry.current_champion is None:
        raise SystemExit(f"No current champion found in {args.registry_path}")

    champion = registry.current_champion
    candidate_policy = champion.policy
    gemini_base = _gemini_policy(
        policy_id="google/gemini-showmatch",
        model=args.gemini_model,
        label="Gemini 3.1 Pro Preview",
    )
    base_template = load_json_document(args.base_template)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_BASE / f"gemini_bo7_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    side_order = ["p1", "p2", "p1", "p2", "p1", "p2", "p1"]
    summary: dict[str, Any] = {
        "created_at": timestamp_utc(),
        "series_type": "best_of_7",
        "candidate": champion.to_dict(),
        "opponent": gemini_base.to_dict(),
        "output_dir": str(output_dir),
        "games": [],
        "candidate_wins": 0,
        "opponent_wins": 0,
        "draws": 0,
        "winner": None,
    }
    summary_path = output_dir / "series_summary.json"

    print(f"Current champion: {champion.champion_id} ({candidate_policy.model})")
    print(f"Opponent: {gemini_base.model}")
    print(f"Output dir: {output_dir}")

    for game_index, candidate_side in enumerate(side_order, start=1):
        if summary["candidate_wins"] >= 4 or summary["opponent_wins"] >= 4:
            break

        trace_seed = args.seed_base + game_index - 1
        game_root = output_dir / f"game_{game_index:02d}_{candidate_side}"
        game_root.mkdir(parents=True, exist_ok=True)
        config_path = game_root / "match_config.json"

        candidate_game_policy = _clone_policy(
            candidate_policy,
            policy_id=f"showmatch/champion-{candidate_side}",
            label=champion.champion_id,
        )
        gemini_side = "p2" if candidate_side == "p1" else "p1"
        gemini_game_policy = _clone_policy(
            gemini_base,
            policy_id=f"showmatch/gemini-{gemini_side}",
            label="Gemini 3.1 Pro Preview",
        )
        if candidate_side == "p1":
            p1_policy = candidate_game_policy
            p2_policy = gemini_game_policy
        else:
            p1_policy = gemini_game_policy
            p2_policy = candidate_game_policy

        config_payload = build_head_to_head_config(
            base_template,
            p1_policy=p1_policy,
            p2_policy=p2_policy,
        )
        config_path.write_text(json.dumps(config_payload, indent=2) + "\n")

        cmd = [
            str(REPO_ROOT / "scripts" / "run_match.sh"),
            "--daemon-config",
            str(config_path),
            "--runs-root",
            str(game_root),
            "--trace-seed",
            str(trace_seed),
            "--log-level",
            args.log_level,
        ]
        if args.skip_mod_push:
            cmd.append("--skip-mod-push")

        print(f"\n[game {game_index}] candidate_side={candidate_side} seed={trace_seed}")
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"run_match.sh failed for game {game_index} with exit code {result.returncode}")

        result_path, payload = _latest_completed_or_failed_result(game_root)
        run_dir = result_path.parent
        replay_video = run_dir / "replay.mp4"
        if not replay_video.is_file():
            raise RuntimeError(
                f"Missing replay video for game {game_index}: expected {replay_video}. "
                "Showmatch requires replay.mp4 for downstream editing."
            )

        outcome = _candidate_outcome(payload, candidate_side=candidate_side)
        if outcome == "win":
            summary["candidate_wins"] += 1
        elif outcome == "loss":
            summary["opponent_wins"] += 1
        else:
            summary["draws"] += 1

        summary["games"].append(
            {
                "game_index": game_index,
                "trace_seed": trace_seed,
                "candidate_side": candidate_side,
                "run_dir": str(run_dir),
                "result_path": str(result_path),
                "replay_video": str(replay_video),
                "match_replay": str(run_dir / "match.replay") if (run_dir / "match.replay").is_file() else None,
                "winner": payload.get("winner"),
                "end_reason": payload.get("end_reason"),
                "total_turns": payload.get("total_turns"),
                "outcome": outcome,
            }
        )
        if summary["candidate_wins"] >= 4:
            summary["winner"] = "candidate"
        elif summary["opponent_wins"] >= 4:
            summary["winner"] = "opponent"

        _write_json(summary_path, summary)
        print(
            f"  series score: candidate {summary['candidate_wins']} - {summary['opponent_wins']} opponent"
            f" (draws={summary['draws']})"
        )

    if summary["winner"] is None:
        if summary["candidate_wins"] > summary["opponent_wins"]:
            summary["winner"] = "candidate"
        elif summary["opponent_wins"] > summary["candidate_wins"]:
            summary["winner"] = "opponent"
        else:
            summary["winner"] = "tied"
    summary["completed_at"] = timestamp_utc()
    _write_json(summary_path, summary)

    print("\nShowmatch complete.")
    print(f"Series summary: {summary_path}")
    print(
        f"Final score: candidate {summary['candidate_wins']} - {summary['opponent_wins']} opponent"
        f" (draws={summary['draws']})"
    )
    print(f"Winner: {summary['winner']}")


if __name__ == "__main__":
    main()
