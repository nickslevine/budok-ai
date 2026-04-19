"""Evaluate a candidate policy against an RL v3 benchmark suite.

Examples:
    uv run --project daemon python scripts/rl_eval_checkpoint_suite.py \
        --model tinker://.../sampler_weights/rl-selfplay-iter10

    uv run --project daemon python scripts/rl_eval_checkpoint_suite.py \
        --model tinker://.../sampler_weights/rl-selfplay-iter10 \
        --suite-path runs/rl_v3/default_suite.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from yomi_daemon.rl.suite_eval import (
    build_policy_from_cli,
    evaluate_suite,
    resolve_model_for_evaluation,
)
from yomi_daemon.rl.v3 import (
    build_default_suite,
    build_suite_seed_bank,
    load_champion_registry,
    load_suite_config,
    load_suite_seed_bank,
    write_suite_seed_bank,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_TEMPLATE = REPO_ROOT / "daemon" / "config" / "rl_self_play_template.json"
DEFAULT_REGISTRY_PATH = REPO_ROOT / "runs" / "rl_v3" / "champion_registry.json"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "runs" / "rl_v3"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate policy against an RL v3 suite"
    )
    parser.add_argument(
        "--model", required=True, help="Model or sampler path for the candidate"
    )
    parser.add_argument("--policy-id", default="rl/candidate")
    parser.add_argument("--label", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--credential-env-var", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--prompt-version", default="strategic_rl_v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--suite-path", default=None, help="Optional suite JSON file")
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--num-matches",
        type=int,
        default=10,
        help="Matches per suite entry when building the default suite",
    )
    parser.add_argument("--history-limit", type=int, default=3)
    parser.add_argument("--no-include-sft", action="store_true")
    parser.add_argument("--no-include-champion", action="store_true")
    parser.add_argument("--base-template", type=Path, default=DEFAULT_BASE_TEMPLATE)
    parser.add_argument("--match-timeout-s", type=int, default=900)
    parser.add_argument(
        "--match-max-attempts",
        type=int,
        default=1,
        help="Retry each suite match up to this many times until a completed run is recorded",
    )
    parser.add_argument(
        "--skip-mod-push",
        action="store_true",
        help="Reuse the currently installed VM mod instead of pushing the local mod before each match",
    )
    parser.add_argument("--seed-base", type=int, default=4000)
    parser.add_argument(
        "--seed-bank-path",
        type=Path,
        default=None,
        help=(
            "Optional seed-bank JSON. If it exists, reuse it. If it does not exist, "
            "the script will generate one from --seed-base and write it there."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    resolved_model = resolve_model_for_evaluation(
        args.model,
        sampler_name=f"{args.policy_id.replace('/', '-')}-eval",
    )

    candidate_policy = build_policy_from_cli(
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
    if args.suite_path:
        suite = load_suite_config(Path(args.suite_path))
    else:
        registry = load_champion_registry(args.registry_path)
        suite = build_default_suite(
            registry=registry,
            num_matches=args.num_matches,
            prompt_version=args.prompt_version,
            include_sft=not args.no_include_sft,
            include_champion=not args.no_include_champion,
            history_limit=args.history_limit,
            exclude_models={candidate_policy.model},
        )

    output_dir = args.output_dir
    if output_dir is None:
        tag = args.tag or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        output_dir = DEFAULT_OUTPUT_BASE / f"suite_eval_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_bank_path = args.seed_bank_path or (output_dir / "seed_bank.json")
    if seed_bank_path.exists():
        seed_bank = load_suite_seed_bank(seed_bank_path)
        print(f"Using existing seed bank: {seed_bank_path}")
    else:
        seed_bank = build_suite_seed_bank(suite, seed_base=args.seed_base)
        write_suite_seed_bank(seed_bank_path, seed_bank)
        print(f"Wrote seed bank: {seed_bank_path}")

    print(f"Candidate: {candidate_policy.model}")
    print(f"Suite entries: {', '.join(entry.entry_id for entry in suite.entries)}")
    print(f"Output dir: {output_dir}")

    result = evaluate_suite(
        repo_root=REPO_ROOT,
        candidate_policy=candidate_policy,
        suite=suite,
        output_root=output_dir,
        base_template_path=args.base_template,
        match_timeout_s=args.match_timeout_s,
        seed_base=args.seed_base,
        seed_bank=seed_bank,
        match_max_attempts=args.match_max_attempts,
        skip_mod_push=args.skip_mod_push,
    )

    result_path = output_dir / "eval_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print()
    print(f"Aggregate score: {result['aggregate_score']:.3f}")
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
