"""Prepare SFT training data from collected match trajectories.

Reads decisions.jsonl files from a runs directory and converts them into
Tinker-compatible training examples: (prompt, completion) pairs where the
prompt is the same text the model sees during live play and the completion
is the action the strong model chose.

Usage:
    uv run python scripts/rl_prepare_sft.py [RUNS_DIR] [--output PATH] [--winners-only]

Arguments:
    RUNS_DIR        Path to runs directory (default: runs/rl_sft_cowboy)

Options:
    --output PATH   Output JSONL path (default: runs/rl_sft_cowboy/sft_training_data.jsonl)
    --winners-only  Only include decisions from the winning side of each match
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare_sft_data(
    runs_dir: Path,
    *,
    winners_only: bool = False,
) -> list[dict]:
    """Extract training examples from match artifacts.

    Each example is a dict with:
        - prompt: the full prompt text the model received
        - completion: the JSON action the model produced
        - metadata: match/turn identifiers
    """
    examples: list[dict] = []
    matches_processed = 0
    skipped_fallbacks = 0
    skipped_losers = 0

    for match_dir in sorted(runs_dir.iterdir()):
        if not match_dir.is_dir():
            continue

        result_file = match_dir / "result.json"
        decisions_file = match_dir / "decisions.jsonl"
        prompts_file = match_dir / "prompts.jsonl"

        if not result_file.exists() or not decisions_file.exists() or not prompts_file.exists():
            continue

        result = json.loads(result_file.read_text())
        if result.get("status") != "completed":
            continue

        winner = result.get("winner")
        matches_processed += 1

        # Index prompts by (player_id, turn_id) for lookup
        prompts_by_key: dict[tuple[str, int], dict] = {}
        with open(prompts_file) as f:
            for line in f:
                p = json.loads(line)
                key = (p.get("player_id"), p.get("turn_id"))
                prompts_by_key[key] = p

        # Process each decision
        with open(decisions_file) as f:
            for line in f:
                dec = json.loads(line)
                rp = dec.get("request_payload", {})
                dp = dec.get("decision_payload", {})

                player_id = rp.get("player_id")
                turn_id = rp.get("turn_id")

                # Skip fallback decisions -- we only want intentional model outputs
                if dp.get("fallback_reason") is not None:
                    skipped_fallbacks += 1
                    continue

                # Optionally skip losing side
                if winners_only and player_id != winner:
                    skipped_losers += 1
                    continue

                # Get the prompt text for this turn
                prompt_data = prompts_by_key.get((player_id, turn_id))
                if prompt_data is None:
                    continue

                prompt_text = prompt_data.get("prompt_text", "")
                if not prompt_text:
                    continue

                # Build completion from the decision
                completion_obj: dict = {"action": dp.get("action")}
                if dp.get("data") is not None:
                    completion_obj["data"] = dp["data"]
                extra = dp.get("extra", {})
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

                completion = json.dumps(completion_obj, separators=(",", ":"))

                examples.append({
                    "prompt": prompt_text,
                    "completion": completion,
                    "metadata": {
                        "match_id": rp.get("match_id"),
                        "turn_id": turn_id,
                        "player_id": player_id,
                        "state_hash": rp.get("state_hash"),
                        "action": dp.get("action"),
                        "winner": winner,
                        "is_winner": player_id == winner,
                    },
                })

    print(f"Processed {matches_processed} matches")
    print(f"Training examples: {len(examples)}")
    print(f"Skipped (fallback): {skipped_fallbacks}")
    if winners_only:
        print(f"Skipped (losing side): {skipped_losers}")

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT training data from match trajectories")
    parser.add_argument(
        "runs_dir",
        nargs="?",
        type=Path,
        default=Path("runs/rl_sft_cowboy"),
        help="Path to runs directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--winners-only",
        action="store_true",
        help="Only include decisions from the winning side",
    )
    args = parser.parse_args()

    output_path = args.output or (args.runs_dir / "sft_training_data.jsonl")

    examples = prepare_sft_data(args.runs_dir, winners_only=args.winners_only)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")

    print(f"\nWrote {len(examples)} examples to {output_path}")

    # Print stats
    if examples:
        avg_prompt_len = sum(len(e["prompt"]) for e in examples) / len(examples)
        avg_completion_len = sum(len(e["completion"]) for e in examples) / len(examples)
        winner_count = sum(1 for e in examples if e["metadata"]["is_winner"])
        print(f"  From winning side: {winner_count}")
        print(f"  From losing side: {len(examples) - winner_count}")
        print(f"  Avg prompt length: {avg_prompt_len:.0f} chars")
        print(f"  Avg completion length: {avg_completion_len:.0f} chars")


if __name__ == "__main__":
    main()
