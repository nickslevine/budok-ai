"""Build an RL v3 league JSON from the current champion registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from yomi_daemon.rl.v3 import (
    build_default_league,
    load_champion_registry,
    write_league_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "runs" / "rl_v3" / "champion_registry.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "runs" / "rl_v3" / "league.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an RL v3 league JSON")
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--prompt-version", default="strategic_rl_v1")
    parser.add_argument("--history-limit", type=int, default=3)
    parser.add_argument("--mirror-weight", type=float, default=0.2)
    parser.add_argument("--champion-weight", type=float, default=0.35)
    parser.add_argument("--sft-weight", type=float, default=0.25)
    parser.add_argument("--history-weight", type=float, default=0.2)
    parser.add_argument("--no-include-sft", action="store_true")
    parser.add_argument("--no-include-mirror", action="store_true")
    args = parser.parse_args()

    registry = load_champion_registry(args.registry_path)
    league = build_default_league(
        registry=registry,
        include_sft=not args.no_include_sft,
        include_mirror=not args.no_include_mirror,
        history_limit=args.history_limit,
        mirror_weight=args.mirror_weight,
        champion_weight=args.champion_weight,
        sft_weight=args.sft_weight,
        history_weight=args.history_weight,
        prompt_version=args.prompt_version,
    )
    write_league_config(args.output, league)
    print(f"Wrote league with {len(league.entries)} entries to {args.output}")
    print(
        "Entries: "
        + ", ".join(f"{entry.entry_id}({entry.weight:.2f})" for entry in league.entries)
    )


if __name__ == "__main__":
    main()
