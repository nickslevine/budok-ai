"""Evaluate a challenger policy and optionally promote it to champion.

This script compares a challenger against the current champion using:
- a head-to-head suite against the current champion, and
- a common benchmark panel (SFT + historical champions, excluding challenger/champion).

If the challenger clears the configured thresholds, the champion registry is
updated in-place.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from yomi_daemon.rl.suite_eval import (
    build_policy_from_cli,
    evaluate_suite,
    resolve_model_for_evaluation,
)
from yomi_daemon.rl.v3 import (
    ChampionRecord,
    ChampionRegistry,
    SuiteConfig,
    SuiteEntry,
    aggregate_suite_score,
    build_default_suite,
    build_suite_seed_bank,
    clone_policy,
    load_champion_registry,
    timestamp_utc,
    write_champion_registry,
    write_suite_seed_bank,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "runs" / "rl_v3" / "champion_registry.json"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "runs" / "rl_v3"
DEFAULT_BASE_TEMPLATE = REPO_ROOT / "daemon" / "config" / "rl_self_play_template.json"


def max_common_panel_regression(
    challenger_result: dict[str, Any],
    champion_result: dict[str, Any],
) -> float:
    challenger_entries = {
        entry["entry_id"]: entry
        for entry in challenger_result.get("entries", [])
        if isinstance(entry, dict)
    }
    champion_entries = {
        entry["entry_id"]: entry
        for entry in champion_result.get("entries", [])
        if isinstance(entry, dict)
    }
    regressions: list[float] = []
    for entry_id, champion_entry in champion_entries.items():
        challenger_entry = challenger_entries.get(entry_id)
        if challenger_entry is None:
            continue
        champion_win_rate = float(
            champion_entry.get("aggregate", {}).get("win_rate", 0.0)
        )
        challenger_win_rate = float(
            challenger_entry.get("aggregate", {}).get("win_rate", 0.0)
        )
        regressions.append(max(0.0, champion_win_rate - challenger_win_rate))
    return max(regressions) if regressions else 0.0


def build_head_to_head_suite(
    *,
    champion_policy,
    champion_id: str,
    num_matches: int,
) -> SuiteConfig:
    return SuiteConfig(
        entries=(
            SuiteEntry(
                entry_id="head_to_head",
                opponent=clone_policy(
                    champion_policy, policy_id="rl/champion-opponent"
                ),
                num_matches=num_matches,
                sides="both",
                weight=1.0,
                label=champion_id,
            ),
        )
    )


def maybe_add_history(
    history: tuple[ChampionRecord, ...], record: ChampionRecord
) -> tuple[ChampionRecord, ...]:
    if any(existing.champion_id == record.champion_id for existing in history):
        return history
    return (record,) + history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a challenger policy to champion if it wins the RL v3 gates"
    )
    parser.add_argument(
        "--model", required=True, help="Model or sampler path for the challenger"
    )
    parser.add_argument("--policy-id", default="rl/challenger")
    parser.add_argument("--label", default=None)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--credential-env-var", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--prompt-version", default="strategic_rl_v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--history-limit", type=int, default=3)
    parser.add_argument("--panel-num-matches", type=int, default=10)
    parser.add_argument("--head-to-head-matches", type=int, default=10)
    parser.add_argument("--min-head-to-head-win-rate", type=float, default=0.5)
    parser.add_argument("--min-panel-delta", type=float, default=0.0)
    parser.add_argument("--max-panel-regression", type=float, default=0.05)
    parser.add_argument("--base-template", type=Path, default=DEFAULT_BASE_TEMPLATE)
    parser.add_argument("--match-timeout-s", type=int, default=900)
    parser.add_argument(
        "--match-max-attempts",
        type=int,
        default=1,
        help="Retry each benchmark match up to this many times until a completed run is recorded",
    )
    parser.add_argument(
        "--skip-mod-push",
        action="store_true",
        help="Reuse the currently installed VM mod instead of pushing the local mod before each benchmark match",
    )
    parser.add_argument("--seed-base", type=int, default=6000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.credential_env_var is None:
        args.credential_env_var = (
            "TINKER_API_KEY"
            if args.model.startswith("tinker://")
            else "OPENROUTER_API_KEY"
        )

    resolved_model = resolve_model_for_evaluation(
        args.model,
        sampler_name=f"{args.policy_id.replace('/', '-')}-promotion",
    )

    challenger_policy = build_policy_from_cli(
        policy_id=args.policy_id,
        model=resolved_model,
        prompt_version=args.prompt_version,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        label=args.label,
        provider=args.provider,
        credential_env_var=args.credential_env_var,
        base_url=args.base_url,
    )
    registry = load_champion_registry(args.registry_path)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            DEFAULT_OUTPUT_BASE
            / f"promotion_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if registry.current_champion is None:
        print("No current champion found; bootstrapping challenger as champion.")
        bootstrap_seed_bank_path = output_dir / "bootstrap_seed_bank.json"
        try:
            bootstrap_suite = build_default_suite(
                registry=registry,
                num_matches=args.panel_num_matches,
                prompt_version=args.prompt_version,
                include_champion=False,
                history_limit=args.history_limit,
                exclude_models={challenger_policy.model},
            )
        except ValueError as exc:
            print(f"Bootstrap benchmark skipped: {exc}")
            bootstrap_result = {
                "skipped": True,
                "reason": str(exc),
                "candidate": challenger_policy.to_dict(),
            }
            bootstrap_score = 0.0
        else:
            bootstrap_seed_bank = build_suite_seed_bank(
                bootstrap_suite, seed_base=args.seed_base
            )
            write_suite_seed_bank(bootstrap_seed_bank_path, bootstrap_seed_bank)
            bootstrap_result = evaluate_suite(
                repo_root=REPO_ROOT,
                candidate_policy=challenger_policy,
                suite=bootstrap_suite,
                output_root=output_dir / "bootstrap_panel",
                base_template_path=args.base_template,
                match_timeout_s=args.match_timeout_s,
                seed_base=args.seed_base,
                seed_bank=bootstrap_seed_bank,
                match_max_attempts=args.match_max_attempts,
                skip_mod_push=args.skip_mod_push,
            )
            bootstrap_score = aggregate_suite_score(bootstrap_result)
        new_record = ChampionRecord(
            champion_id=args.label
            or f"champion-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
            policy=challenger_policy,
            promoted_at=timestamp_utc(),
            benchmark_summary={
                "bootstrap_panel": bootstrap_result,
                "aggregate_score": bootstrap_score,
            },
        )
        new_registry = ChampionRegistry(
            current_champion=new_record,
            history=registry.history,
            promotions=registry.promotions
            + (
                {
                    "promoted_at": new_record.promoted_at,
                    "reason": "bootstrap",
                    "new_champion": new_record.to_dict(),
                },
            ),
        )
        comparison: dict[str, Any] = {
            "bootstrap": True,
            "bootstrap_result": bootstrap_result,
        }
        if bootstrap_seed_bank_path.exists():
            comparison["bootstrap_seed_bank_path"] = str(bootstrap_seed_bank_path)
        (output_dir / "promotion_result.json").write_text(
            json.dumps(comparison, indent=2) + "\n"
        )
        if args.dry_run:
            print("Dry run: not writing champion registry.")
        else:
            write_champion_registry(args.registry_path, new_registry)
            print(f"Wrote champion registry to {args.registry_path}")
        return

    current = registry.current_champion
    assert current is not None
    champion_policy = clone_policy(
        current.policy,
        policy_id="rl/current-champion",
        prompt_version=args.prompt_version,
    )

    try:
        panel_suite = build_default_suite(
            registry=registry,
            num_matches=args.panel_num_matches,
            prompt_version=args.prompt_version,
            include_champion=False,
            history_limit=args.history_limit,
            exclude_models={challenger_policy.model, champion_policy.model},
        )
    except ValueError as exc:
        print(f"Common panel skipped: {exc}")
        panel_suite = None
        challenger_panel_result = {
            "skipped": True,
            "reason": str(exc),
        }
        champion_panel_result = {
            "skipped": True,
            "reason": str(exc),
        }
        panel_seed_bank_path = None
    else:
        panel_seed_bank = build_suite_seed_bank(panel_suite, seed_base=args.seed_base)
        panel_seed_bank_path = output_dir / "panel_seed_bank.json"
        write_suite_seed_bank(panel_seed_bank_path, panel_seed_bank)

        print(f"Evaluating challenger panel at {output_dir / 'challenger_panel'}")
        challenger_panel_result = evaluate_suite(
            repo_root=REPO_ROOT,
            candidate_policy=challenger_policy,
            suite=panel_suite,
            output_root=output_dir / "challenger_panel",
            base_template_path=args.base_template,
            match_timeout_s=args.match_timeout_s,
            seed_base=args.seed_base,
            seed_bank=panel_seed_bank,
            match_max_attempts=args.match_max_attempts,
            skip_mod_push=args.skip_mod_push,
        )
        print(f"Evaluating champion panel at {output_dir / 'champion_panel'}")
        champion_panel_result = evaluate_suite(
            repo_root=REPO_ROOT,
            candidate_policy=champion_policy,
            suite=panel_suite,
            output_root=output_dir / "champion_panel",
            base_template_path=args.base_template,
            match_timeout_s=args.match_timeout_s,
            seed_base=args.seed_base,
            seed_bank=panel_seed_bank,
            match_max_attempts=args.match_max_attempts,
            skip_mod_push=args.skip_mod_push,
        )

    head_to_head_suite = build_head_to_head_suite(
        champion_policy=champion_policy,
        champion_id=current.champion_id,
        num_matches=args.head_to_head_matches,
    )
    head_to_head_seed_bank = build_suite_seed_bank(
        head_to_head_suite, seed_base=args.seed_base + 100_000
    )
    head_to_head_seed_bank_path = output_dir / "head_to_head_seed_bank.json"
    write_suite_seed_bank(head_to_head_seed_bank_path, head_to_head_seed_bank)

    print(f"Evaluating challenger head-to-head at {output_dir / 'head_to_head'}")
    challenger_head_result = evaluate_suite(
        repo_root=REPO_ROOT,
        candidate_policy=challenger_policy,
        suite=head_to_head_suite,
        output_root=output_dir / "head_to_head",
        base_template_path=args.base_template,
        match_timeout_s=args.match_timeout_s,
        seed_base=args.seed_base + 100_000,
        seed_bank=head_to_head_seed_bank,
        match_max_attempts=args.match_max_attempts,
        skip_mod_push=args.skip_mod_push,
    )

    challenger_panel_score = aggregate_suite_score(challenger_panel_result)
    champion_panel_score = aggregate_suite_score(champion_panel_result)
    head_to_head_entry = challenger_head_result["entries"][0]
    head_to_head_win_rate = float(head_to_head_entry["aggregate"]["win_rate"])
    panel_delta = challenger_panel_score - champion_panel_score
    max_regression = max_common_panel_regression(
        challenger_panel_result, champion_panel_result
    )
    should_promote = (
        head_to_head_win_rate > args.min_head_to_head_win_rate
        and panel_delta >= args.min_panel_delta
        and max_regression <= args.max_panel_regression
    )

    comparison: dict[str, Any] = {
        "evaluated_at": timestamp_utc(),
        "challenger": challenger_policy.to_dict(),
        "current_champion": current.to_dict(),
        "challenger_panel_score": challenger_panel_score,
        "champion_panel_score": champion_panel_score,
        "panel_delta": panel_delta,
        "head_to_head_win_rate": head_to_head_win_rate,
        "max_common_panel_regression": max_regression,
        "thresholds": {
            "min_head_to_head_win_rate": args.min_head_to_head_win_rate,
            "min_panel_delta": args.min_panel_delta,
            "max_panel_regression": args.max_panel_regression,
        },
        "head_to_head_seed_bank_path": str(head_to_head_seed_bank_path),
        "should_promote": should_promote,
        "challenger_panel_result": challenger_panel_result,
        "champion_panel_result": champion_panel_result,
        "challenger_head_to_head_result": challenger_head_result,
    }
    if panel_seed_bank_path is not None:
        comparison["panel_seed_bank_path"] = str(panel_seed_bank_path)
    comparison_path = output_dir / "promotion_result.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")

    print()
    print(f"head_to_head_win_rate: {head_to_head_win_rate:.1%}")
    print(f"panel_delta: {panel_delta:+.3f}")
    print(f"max_common_panel_regression: {max_regression:.1%}")
    print(
        f"Promotion decision: {'PROMOTE' if should_promote else 'KEEP CURRENT CHAMPION'}"
    )
    print(f"Wrote comparison to {comparison_path}")

    if should_promote and not args.dry_run:
        new_record = ChampionRecord(
            champion_id=args.label
            or f"champion-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
            policy=challenger_policy,
            promoted_at=timestamp_utc(),
            benchmark_summary={
                "panel_score": challenger_panel_score,
                "head_to_head_win_rate": head_to_head_win_rate,
                "comparison_path": str(comparison_path),
            },
        )
        new_history = maybe_add_history(registry.history, current)
        new_registry = ChampionRegistry(
            current_champion=new_record,
            history=new_history,
            promotions=registry.promotions
            + (
                {
                    "promoted_at": new_record.promoted_at,
                    "old_champion_id": current.champion_id,
                    "new_champion_id": new_record.champion_id,
                    "comparison_path": str(comparison_path),
                    "panel_delta": panel_delta,
                    "head_to_head_win_rate": head_to_head_win_rate,
                    "max_common_panel_regression": max_regression,
                },
            ),
        )
        write_champion_registry(args.registry_path, new_registry)
        print(f"Updated champion registry at {args.registry_path}")
    elif args.dry_run:
        print("Dry run: not writing champion registry.")


if __name__ == "__main__":
    main()
