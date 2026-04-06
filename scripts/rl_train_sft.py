"""SFT training on Tinker using collected GPT-5.4 match data.

Trains Qwen3-8B with LoRA on (prompt, completion) pairs from
runs/rl_sft_cowboy/sft_training_data.jsonl.

Usage:
    # Load .env first, then run:
    set -a && source .env && set +a
    uv run python scripts/rl_train_sft.py [OPTIONS]

Options:
    --data PATH          Training data JSONL (default: runs/rl_sft_cowboy/sft_training_data.jsonl)
    --model MODEL        Tinker model ID (default: Qwen/Qwen3-8B)
    --epochs N           Number of epochs (default: 3)
    --batch-size N       Examples per batch (default: 4)
    --lr FLOAT           Learning rate (default: 0.0002)
    --rank N             LoRA rank (default: 16)
    --max-length N       Max token length per example (default: 8192)
    --save-name NAME     Checkpoint name (default: yomi-sft-cowboy-v1)
    --eval-samples N     Held-out samples for eval (default: 50)
    --dry-run            Load data and print stats without training
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# Force unbuffered output so we can monitor progress
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


async def train(args: argparse.Namespace) -> None:
    import tinker
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    # ── Load training data ──────────────────────────────────────────────
    data_path = Path(args.data)
    print(f"Loading training data from {data_path}")
    raw_examples = []
    with open(data_path) as f:
        for line in f:
            raw_examples.append(json.loads(line))

    random.seed(42)
    random.shuffle(raw_examples)

    # Split eval set
    eval_examples = raw_examples[: args.eval_samples]
    train_examples = raw_examples[args.eval_samples :]
    print(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    # ── Create training client ──────────────────────────────────────────
    print(f"\nConnecting to Tinker ({args.model}, rank={args.rank})...")
    service = tinker.ServiceClient()
    tc = await service.create_lora_training_client_async(
        base_model=args.model,
        rank=args.rank,
    )
    tokenizer = tc.get_tokenizer()

    # Qwen3 renderer for chat template
    renderer = get_renderer("qwen3", tokenizer)
    print("Training client ready")

    # ── Tokenize data ───────────────────────────────────────────────────
    print(f"\nTokenizing {len(train_examples)} training examples (max_length={args.max_length})...")
    train_data = []
    skipped = 0
    for ex in train_examples:
        conv = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["completion"]},
        ]
        try:
            datum = conversation_to_datum(
                conv,
                renderer,
                max_length=args.max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            train_data.append(datum)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skipped example: {e}")

    eval_data = []
    for ex in eval_examples:
        conv = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["completion"]},
        ]
        try:
            datum = conversation_to_datum(
                conv,
                renderer,
                max_length=args.max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            eval_data.append(datum)
        except Exception:
            pass

    print(f"Tokenized: {len(train_data)} train, {len(eval_data)} eval ({skipped} skipped)")

    if not train_data:
        print("ERROR: No training data after tokenization")
        return

    # Token stats
    total_tokens = sum(d.model_input.length for d in train_data)
    avg_tokens = total_tokens / len(train_data)
    print(f"Avg tokens/example: {avg_tokens:.0f}, total: {total_tokens:,}")

    if args.dry_run:
        est_tokens_per_epoch = total_tokens
        est_cost = est_tokens_per_epoch * args.epochs * 0.40 / 1_000_000
        print(f"\nDRY RUN - Estimated cost: ${est_cost:.2f} for {args.epochs} epochs")
        return

    # ── Training loop ───────────────────────────────────────────────────
    num_batches = (len(train_data) + args.batch_size - 1) // args.batch_size
    total_steps = num_batches * args.epochs
    print(f"\nTraining: {args.epochs} epochs, {num_batches} batches/epoch, {total_steps} total steps")
    print(f"Learning rate: {args.lr}, batch size: {args.batch_size}")
    print()

    step = 0
    for epoch in range(args.epochs):
        # Shuffle training data each epoch
        indices = list(range(len(train_data)))
        random.shuffle(indices)

        epoch_losses = []
        epoch_start = time.time()

        for batch_idx in range(num_batches):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(train_data))
            batch = [train_data[indices[i]] for i in range(batch_start, batch_end)]

            t0 = time.time()

            # Submit forward/backward and optimizer step
            fwd_future = await tc.forward_backward_async(batch, "cross_entropy")
            opt_future = await tc.optim_step_async(
                tinker.AdamParams(learning_rate=args.lr)
            )

            fwd_result = await fwd_future.result_async()
            await opt_future.result_async()

            elapsed = time.time() - t0

            # Compute loss
            logprobs = np.concatenate(
                [out["logprobs"].tolist() for out in fwd_result.loss_fn_outputs]
            )
            weights = np.concatenate(
                [d.loss_fn_inputs["weights"].tolist() for d in batch]
            )
            loss = -np.dot(logprobs, weights) / max(weights.sum(), 1e-8)
            epoch_losses.append(loss)

            step += 1
            if step % 10 == 0 or step == 1:
                print(
                    f"  epoch {epoch + 1}/{args.epochs} "
                    f"step {step}/{total_steps} "
                    f"batch {batch_idx + 1}/{num_batches} "
                    f"loss={loss:.4f} "
                    f"({elapsed:.1f}s)"
                )

        epoch_elapsed = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        print(
            f"\n  Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f} "
            f"({epoch_elapsed:.0f}s, {epoch_elapsed / 60:.1f}min)"
        )

        # Eval loss
        if eval_data:
            eval_losses = []
            for i in range(0, len(eval_data), args.batch_size):
                batch = eval_data[i : i + args.batch_size]
                fwd = await tc.forward_backward_async(batch, "cross_entropy")
                # No optim_step for eval
                result = await fwd.result_async()
                logprobs = np.concatenate(
                    [out["logprobs"].tolist() for out in result.loss_fn_outputs]
                )
                weights = np.concatenate(
                    [d.loss_fn_inputs["weights"].tolist() for d in batch]
                )
                eval_loss = -np.dot(logprobs, weights) / max(weights.sum(), 1e-8)
                eval_losses.append(eval_loss)
            print(f"  Eval loss: {np.mean(eval_losses):.4f}")

        print()

    # ── Save checkpoint ─────────────────────────────────────────────────
    print(f"Saving checkpoint as '{args.save_name}'...")
    sc = await tc.save_weights_and_get_sampling_client_async(name=args.save_name)
    print(f"Checkpoint saved. Sampling client ready.")

    # Quick sanity check: generate one action
    print("\nSanity check - generating action for first eval example...")
    if eval_examples:
        test_prompt = eval_examples[0]["prompt"]
        stop_sequences = renderer.get_stop_sequences()
        params = tinker.SamplingParams(
            max_tokens=200,
            temperature=0.7,
            stop=stop_sequences,
        )
        messages = [{"role": "user", "content": test_prompt}]
        prompt_tokens = renderer.build_generation_prompt(messages)
        result = await sc.sample_async(
            prompt=prompt_tokens,
            num_samples=1,
            sampling_params=params,
        )
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        print(f"Model output: {response}")

    print("\nDone!")


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training on Tinker")
    parser.add_argument(
        "--data",
        type=str,
        default="runs/rl_sft_cowboy/sft_training_data.jsonl",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--save-name", default="yomi-sft-cowboy-v1")
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
