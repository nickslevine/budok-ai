"""Prepare SFT training data from collected match trajectories.

Reads decisions.jsonl files from a runs directory and converts them into
Tinker-compatible training examples: (prompt, completion) pairs where the
prompt is the same text the model sees during live play and the completion
is the action the strong model chose.

Usage:
    uv run --project daemon python scripts/rl_prepare_sft.py \
        [RUNS_DIR] [--output PATH] [--winners-only] [--strip-reasoning]

Arguments:
    RUNS_DIR        Path to runs directory (default: runs/rl_sft_cowboy)

Options:
    --output PATH         Output JSONL path
    --winners-only        Only include decisions from the winning side
    --strip-reasoning     Omit reasoning from the completion target
    --prompt-version VER  Re-render prompts with this prompt version instead of
                          reusing recorded prompt text from prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yomi_daemon.protocol import (
    ActionDecision,
    DIVector,
    DecisionExtras,
    DecisionRequest,
)
from yomi_daemon.rl.tinker_env import format_completion_from_decision, format_prompt


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
    )


def format_completion_from_payload(
    request_payload: dict,
    decision_payload: dict,
    *,
    strip_reasoning: bool,
) -> str:
    """Build the canonical completion string for a rollout decision."""
    decision = build_action_decision(request_payload, decision_payload)
    completion = format_completion_from_decision(decision)
    if strip_reasoning or not decision.reasoning:
        return completion

    completion_obj = json.loads(completion)
    completion_obj["reasoning"] = decision.reasoning
    return json.dumps(completion_obj, separators=(",", ":"))


def render_prompt_from_request_payload(
    request_payload: dict,
    *,
    prompt_version: str,
) -> str:
    """Re-render a prompt from the structured request payload."""
    request = DecisionRequest.from_dict(request_payload, context="request_payload")
    return format_prompt(request, prompt_version=prompt_version)


def prepare_sft_data(
    runs_dir: Path,
    *,
    winners_only: bool = False,
    strip_reasoning: bool = False,
    prompt_version: str | None = None,
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
    skipped_prompt_rerenders = 0

    for match_dir in sorted(runs_dir.iterdir()):
        if not match_dir.is_dir():
            continue

        result_file = match_dir / "result.json"
        decisions_file = match_dir / "decisions.jsonl"
        prompts_file = match_dir / "prompts.jsonl"

        if not result_file.exists() or not decisions_file.exists():
            continue
        if prompt_version is None and not prompts_file.exists():
            continue

        result = json.loads(result_file.read_text())
        if result.get("status") != "completed":
            continue

        winner = result.get("winner")
        matches_processed += 1

        prompts_by_key: dict[tuple[str, int], str] = {}
        if prompt_version is None:
            with open(prompts_file) as f:
                for line in f:
                    p = json.loads(line)
                    key = (p.get("player_id"), p.get("turn_id"))
                    if isinstance(key[0], str) and isinstance(key[1], int):
                        prompts_by_key[key] = p.get("prompt_text", "")

        with open(decisions_file) as f:
            for line in f:
                dec = json.loads(line)
                rp = dec.get("request_payload", {})
                dp = dec.get("decision_payload", {})

                player_id = rp.get("player_id")
                turn_id = rp.get("turn_id")

                if dp.get("fallback_reason") is not None:
                    skipped_fallbacks += 1
                    continue

                if winners_only and player_id != winner:
                    skipped_losers += 1
                    continue

                if prompt_version is None:
                    if not isinstance(player_id, str) or not isinstance(turn_id, int):
                        continue
                    prompt_text = prompts_by_key.get((player_id, turn_id), "")
                else:
                    try:
                        prompt_text = render_prompt_from_request_payload(
                            rp,
                            prompt_version=prompt_version,
                        )
                    except Exception:
                        skipped_prompt_rerenders += 1
                        continue

                if not prompt_text:
                    continue

                try:
                    completion = format_completion_from_payload(
                        rp,
                        dp,
                        strip_reasoning=strip_reasoning,
                    )
                except Exception:
                    continue

                examples.append(
                    {
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
                    }
                )

    print(f"Processed {matches_processed} matches")
    print(f"Training examples: {len(examples)}")
    print(f"Skipped (fallback): {skipped_fallbacks}")
    if winners_only:
        print(f"Skipped (losing side): {skipped_losers}")
    if prompt_version is not None:
        print(f"Skipped (prompt re-render failure): {skipped_prompt_rerenders}")

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SFT training data from match trajectories"
    )
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
    parser.add_argument(
        "--strip-reasoning",
        action="store_true",
        help="Omit reasoning from the completion target",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Re-render prompts with this prompt version instead of using recorded prompt text",
    )
    args = parser.parse_args()

    default_name = (
        "sft_training_data_action_only.jsonl"
        if args.strip_reasoning
        else "sft_training_data.jsonl"
    )
    output_path = args.output or (args.runs_dir / default_name)

    examples = prepare_sft_data(
        args.runs_dir,
        winners_only=args.winners_only,
        strip_reasoning=args.strip_reasoning,
        prompt_version=args.prompt_version,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")

    print(f"\nWrote {len(examples)} examples to {output_path}")

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
