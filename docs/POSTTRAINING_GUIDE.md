# Post-Training Guide: SFT + RL for YOMI Hustle

## Goal

Train a small LLM (7-8B) via SFT and RL to consistently beat Gemini 3.1 Pro at YOMI Hustle in Cowboy mirror matches.

## Architecture Overview

The existing system already frames the game as a text-based decision problem. Each turn, the model receives a structured game state prompt and selects a named action from a variable legal action set. This is a single-turn architecture -- each prompt is independent with selected context from previous rounds -- which maps directly to GRPO/SFT training without multi-turn credit assignment.

```
Training data flow:

  Godot game ──WebSocket──> Daemon ──> PolicyAdapter ��─> LLM
       ▲                                    │
       └────── ActionDecision ◄───────���─────┘
                    │
                    ▼
              decisions.jsonl ──> Trajectory ──> Training examples
                                                      │
                                                      ▼
                                                 Tinker GRPO
```

Key modules:
- `daemon/src/yomi_daemon/rl/` -- reward functions, trajectory collection, Tinker env wrapper
- `daemon/config/rl_*.json` -- experiment configs
- `scripts/rl_*.sh` -- data collection and evaluation scripts

## Decisions

- **Character:** Cowboy mirror matches for all experiments
- **Starting model:** Qwen 2.5 7B (pending Tinker availability)
- **Reward shape:** Sparse (win/loss only) through Phase 2. Add HP delta shaping only if needed in Phase 3.
- **Training platform:** Tinker (cloud LoRA training, no local GPUs needed)

---

## Phase 0: Collect SFT Training Data

Collect GPT-5.4 vs GPT-5.4 Cowboy mirror matches. Both sides produce strong play; we keep all winning trajectories.

**Estimated cost:** ~$30 for 20 matches (~500 training examples from winning trajectories).

### Run data collection

```bash
# Collect 20 matches (default)
scripts/rl_collect_sft.sh

# Collect a custom number
scripts/rl_collect_sft.sh 40

# Skip mod push if already installed in VM
scripts/rl_collect_sft.sh 20 --skip-mod-push
```

**Output:** `runs/rl_sft_cowboy/<timestamp>_<match_id>/` per match, each containing `decisions.jsonl`, `result.json`, `prompts.jsonl`, etc.

**Config:** `daemon/config/rl_sft_cowboy_mirror.json`
- GPT-5.4 vs GPT-5.4, Cowboy assigned, 750 starting HP, training_room stage

### Verify data quality

After collection, check for failed matches and fallback rates:

```bash
# Count completed vs failed matches
find runs/rl_sft_cowboy -name "result.json" -exec python3 -c "
import json,sys
r = json.load(open(sys.argv[1]))
print(f'{r.get(\"status\",\"?\")}: winner={r.get(\"winner\",\"?\")}, turns={r.get(\"total_turns\",\"?\")}')
" {} \;

# Check fallback rate (should be low for GPT-5.4)
find runs/rl_sft_cowboy -name "decisions.jsonl" -exec grep -c "fallback_reason" {} \; | \
  awk '{s+=$1} END {print "Total fallbacks:", s}'
```

---

## Phase 1: Supervised Fine-Tuning (SFT)

Fine-tune the base model on winning trajectories from Phase 0. **No game time required** -- this is pure supervised learning on collected data.

### Prepare training data

Use the RL module to convert match artifacts into Tinker training examples:

```python
from pathlib import Path
from yomi_daemon.rl.serialization import load_trajectory_records
from yomi_daemon.rl.collector import TrajectoryCollector
from yomi_daemon.rl.rewards import RewardConfig
from yomi_daemon.rl.tinker_env import build_training_example

# Load and filter to winning trajectories, then build training examples
# (Full training data preparation script TBD -- will live in scripts/rl_prepare_sft.py)
```

### Training on Tinker

The training loop uses Tinker's `TrainingClient` with cross-entropy loss on (prompt, completion) pairs from winning games:

```python
import tinker

service = tinker.ServiceClient()
tc = await service.create_lora_training_client(
    base_model="qwen/qwen-2.5-7b",  # check Tinker availability
    rank=32,
    train_mlp=True,
    train_attn=True,
)

for epoch in range(num_epochs):
    for batch in training_data:
        data = [
            tinker.types.Datum(
                model_input=tinker.types.ModelInput.from_ints(
                    tokenizer.encode(example["prompt"] + example["completion"])
                ),
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "weights": weights,  # mask prompt tokens, train on completion only
                },
            )
            for example in batch
        ]
        await tc.forward_backward_async(data, loss_fn="cross_entropy")
        await tc.optim_step_async(tinker.types.AdamParams(learning_rate=1e-4))

# Save weights
await tc.save_state("sft_cowboy_v1")
```

### Evaluate SFT model

```bash
# Evaluate against GPT-5.4 (10 matches)
scripts/rl_eval.sh --model "tinker://path/to/sft_cowboy_v1" \
    --opponent gpt54 --num-matches 10 --tag "sft_v1_vs_gpt54"

# Evaluate against Gemini 3.1 Pro (10 matches)
scripts/rl_eval.sh --model "tinker://path/to/sft_cowboy_v1" \
    --opponent gemini --num-matches 10 --tag "sft_v1_vs_gemini"
```

**Success criteria:**
- Beats `baseline/random` >90%
- Beats `baseline/scripted_safe` >60%
- Produces valid JSON actions with <10% fallback rate

---

## Phase 2: Rejection Sampling / Expert Iteration

Iteratively improve the model by collecting matches with the current policy, keeping only winning trajectories, and SFT-ing again. This is simple, proven, and much cheaper than full RL.

### Loop

```
for each iteration:
    1. Run 200 matches with current model vs scripted baselines
    2. Filter to winning trajectories
    3. SFT on original expert data + new winning trajectories
    4. Evaluate against Gemini/GPT-5.4
    5. If win rate improves, keep; otherwise stop
```

### Run matches with current model

```bash
# Create a config that uses the current model vs baselines
# (will need a config variant or --p1-policy/--p2-policy overrides)

scripts/rl_collect_sft.sh 200 --daemon-config daemon/config/rl_rejection_sampling.json
```

### Evaluate

```bash
scripts/rl_eval.sh --model "tinker://path/to/iter_N" \
    --opponent both --num-matches 10 --tag "iter_N"
```

**Success criteria:** Win rate against Gemini 3.1 Pro improves with each iteration.

---

## Phase 3: GRPO (Only If Phase 2 Plateaus)

Full reinforcement learning with Group Relative Policy Optimization. Each match is a rollout; the match outcome scores all steps in that match.

### Reward configuration

Start sparse, add density only if needed:

```python
from yomi_daemon.rl.rewards import RewardConfig, hp_delta, fallback_penalty

# Start here
config = RewardConfig.sparse()  # win/loss only

# If convergence is too slow, try this
config = RewardConfig(
    components={
        "hp_delta": (hp_delta, 0.1),       # low weight to avoid reward hacking
        "fallback": (fallback_penalty, 0.1),
    },
    terminal_weight=1.0,
)
```

### GRPO training loop (sketch)

```python
for iteration in range(num_iterations):
    # 1. Save current weights for sampling
    sc = await tc.save_weights_and_get_sampling_client(f"grpo_iter_{iteration}")

    # 2. Collect rollouts (run matches with current policy)
    #    Each match produces a trajectory scored by outcome
    trajectories = await collect_rollouts(sc, num_matches=50)

    # 3. Group by match, compute advantages
    #    All steps in a winning match get positive advantage,
    #    all steps in a losing match get negative advantage
    groups = group_by_match(trajectories)
    advantages = compute_group_relative_advantages(groups)

    # 4. Build training data
    data = assemble_training_data(groups, advantages)

    # 5. Train
    await tc.forward_backward_async(data, loss_fn="ppo")
    await tc.optim_step_async(AdamParams(learning_rate=5e-5))

    # 6. Evaluate every N iterations
    if iteration % 10 == 0:
        evaluate(sc)
```

### Self-play (optional extension)

If the model beats scripted baselines but not Gemini:

1. Freeze current best model as opponent
2. Train against frozen opponent
3. Periodically update frozen opponent with latest weights
4. Maintain a league of past checkpoints to avoid strategy cycling

---

## Evaluation

### Quick eval (during development)

```bash
# vs GPT-5.4 only (cheaper, faster)
scripts/rl_eval.sh --model MODEL_ID --opponent gpt54 --num-matches 10 --tag "quick"
```

### Full eval (milestone checkpoints)

```bash
# vs both Gemini and GPT-5.4
scripts/rl_eval.sh --model MODEL_ID --opponent both --num-matches 20 --tag "milestone_v1"
```

### Reading results

Results are serialized to `runs/rl_eval_<tag>/eval_results.json`:

```json
{
  "model": "tinker://path/to/model",
  "timestamp": "2026-04-05T...",
  "total_games": 20,
  "opponents": {
    "google/gemini-3.1-pro-preview": {
      "games": 10, "wins": 3, "losses": 7, "draws": 0,
      "win_rate": 0.3, "avg_hp_diff": -120.0, "avg_turns": 42.0
    }
  },
  "games": [...]
}
```

Key metrics to track across experiments:
- **Win rate** vs each opponent
- **Avg HP differential** (positive = our model ahead at match end)
- **Fallback rate** (should decrease with training)
- **Avg turns** (very short or very long matches may indicate degenerate play)

---

## Clock Time Budget

The binding constraint is match execution time (~30s per match). Budget accordingly:

| Phase | Matches needed | Time (serial) | Time (8 parallel) |
|---|---|---|---|
| Phase 0: SFT data | 20 | ~10 min | ~2 min |
| Phase 1: SFT training | 0 (offline) | 0 | 0 |
| Phase 1: Eval | 20 | ~10 min | ~2 min |
| Phase 2: Per iteration | 200 | ~100 min | ~13 min |
| Phase 2: Eval | 20 | ~10 min | ~2 min |
| Phase 3: Per iteration | 50 | ~25 min | ~4 min |

Mitigations for slow clock time:
1. **Parallel Godot instances** (each on a different port)
2. **`Engine.time_scale`** acceleration in Godot (untested -- may break game logic)
3. **Headless rendering** (`--no-replay` already skips video recording)
4. **750 HP** instead of 1500 (already configured, halves match length)

---

## File Reference

| File | Purpose |
|---|---|
| `daemon/src/yomi_daemon/rl/rewards.py` | Composable reward functions |
| `daemon/src/yomi_daemon/rl/trajectory.py` | Step, Trajectory, MatchOutcome types |
| `daemon/src/yomi_daemon/rl/collector.py` | TrajectoryCollector for live matches |
| `daemon/src/yomi_daemon/rl/tinker_env.py` | Prompt/completion formatting for Tinker |
| `daemon/src/yomi_daemon/rl/serialization.py` | JSONL read/write for trajectories |
| `daemon/src/yomi_daemon/rl/features.py` | Numeric feature extraction from observations |
| `daemon/config/rl_sft_cowboy_mirror.json` | GPT-5.4 mirror config for SFT data |
| `daemon/config/rl_eval_vs_gemini.json` | Eval config (model vs Gemini/GPT-5.4) |
| `scripts/rl_collect_sft.sh` | Run N SFT data collection matches |
| `scripts/rl_eval.sh` | Evaluate model, serialize HP differential results |
| `docs/rl_brainstorming.md` | Extended brainstorming and analysis |
| `docs/rl_libraries_infra.md` | Tinker and OpenReward research |
| `docs/rl_decisions.md` | Key experiment decisions and rationale |
