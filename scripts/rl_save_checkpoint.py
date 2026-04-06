"""Re-run training and save a persistent checkpoint.

Since save_weights_and_get_sampling_client doesn't persist by default,
this script re-trains and uses save_state with a long TTL.

Usage:
    set -a && source .env && set +a
    PYTHONUNBUFFERED=1 uv run --project daemon python scripts/rl_save_checkpoint.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


async def train_and_save() -> None:
    import tinker
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    data_path = Path("runs/rl_sft_cowboy/sft_training_data.jsonl")
    print(f"Loading data from {data_path}")
    raw_examples = []
    with open(data_path) as f:
        for line in f:
            raw_examples.append(json.loads(line))
    random.seed(42)
    random.shuffle(raw_examples)
    train_examples = raw_examples[50:]
    print(f"Train: {len(train_examples)} examples")

    print("Connecting to Tinker...")
    service = tinker.ServiceClient()
    tc = await service.create_lora_training_client_async(
        base_model="Qwen/Qwen3-8B", rank=16
    )
    tokenizer = tc.get_tokenizer()
    renderer = get_renderer("qwen3", tokenizer)

    print("Tokenizing...")
    train_data = []
    for ex in train_examples:
        conv = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["completion"]},
        ]
        try:
            datum = conversation_to_datum(
                conv, renderer, max_length=8192,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            train_data.append(datum)
        except Exception:
            pass
    print(f"Tokenized: {len(train_data)} examples")

    # Train 3 epochs
    batch_size = 4
    num_batches = (len(train_data) + batch_size - 1) // batch_size
    lr = 0.0002

    for epoch in range(3):
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_losses = []
        t0 = time.time()

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(train_data))
            batch = [train_data[indices[i]] for i in range(start, end)]

            fwd = await tc.forward_backward_async(batch, "cross_entropy")
            opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=lr))
            result = await fwd.result_async()
            await opt.result_async()

            logprobs = np.concatenate(
                [out["logprobs"].tolist() for out in result.loss_fn_outputs]
            )
            weights = np.concatenate(
                [d.loss_fn_inputs["weights"].tolist() for d in batch]
            )
            loss = -np.dot(logprobs, weights) / max(weights.sum(), 1e-8)
            epoch_losses.append(loss)

            step = epoch * num_batches + batch_idx + 1
            if step % 50 == 0 or step == 1:
                print(f"  epoch {epoch+1}/3 step {step} loss={loss:.4f}")

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: avg_loss={np.mean(epoch_losses):.4f} ({elapsed:.0f}s)")

    # Save persistent checkpoint with 30-day TTL
    print("\nSaving persistent checkpoint...")
    save_future = await tc.save_state_async(
        name="yomi-sft-cowboy-v1",
        ttl_seconds=30 * 24 * 3600,  # 30 days
    )
    state_path = await save_future.result_async()
    print(f"Checkpoint saved: {state_path}")

    # Also save sampler weights
    print("Saving sampler weights...")
    sampler_future = await tc.save_weights_for_sampler_async(
        name="yomi-sft-cowboy-v1",
    )
    sampler_path = await sampler_future.result_async()
    print(f"Sampler weights: {sampler_path}")

    print("\nDone! Use these paths for inference.")


if __name__ == "__main__":
    asyncio.run(train_and_save())
