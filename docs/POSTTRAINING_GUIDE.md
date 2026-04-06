# Post-Training Guide: SFT + RL for YOMI Hustle

## Goal

Train a small LLM (8B) via SFT and RL to consistently beat Gemini 3.1 Pro at YOMI Hustle in Cowboy mirror matches.

## Results Summary

**SFT alone took Qwen3-8B from 0% to 67% win rate against frontier models.**

| Metric | Baseline (Qwen3-8B) | SFT v1 |
|---|---|---|
| vs Gemini 3.1 Pro | 0W-3L (0%) | **2W-1L (67%)** |
| vs GPT-5.4 | 0W-3L (0%) | **2W-1L (67%)** |
| Avg HP diff vs Gemini | -377 | **+232** |
| Avg HP diff vs GPT-5.4 | -244 | **-12** |
| Fallback rate | 42% | **0%** |
| Training cost | -- | **~$11** |
| Training data | -- | 1,672 examples from 20 GPT-5.4 mirrors |

---

## Architecture Overview

The existing system frames the game as a text-based decision problem. Each turn, the model receives a structured game state prompt and selects a named action from a variable legal action set. This is a single-turn architecture -- each prompt is independent with selected context from previous rounds -- which maps directly to GRPO/SFT training without multi-turn credit assignment.

```
Training data flow:

  Godot game ──WebSocket──> Daemon ──> PolicyAdapter ──> LLM
       ▲                                    │
       └────── ActionDecision ◄─────────────┘
                    │
                    ▼
              decisions.jsonl ──> Trajectory ──> Training examples
                                                      │
                                                      ▼
                                                 Tinker SFT/GRPO
```

Key modules:
- `daemon/src/yomi_daemon/rl/` -- reward functions, trajectory collection, Tinker env wrapper
- `daemon/config/rl_*.json` -- experiment configs
- `scripts/rl_*.sh` -- data collection and evaluation scripts
- `scripts/rl_train_sft.py` -- SFT training on Tinker
- `scripts/rl_prepare_sft.py` -- convert match data to training examples

## Decisions

- **Character:** Cowboy mirror matches for all experiments
- **Model:** Qwen3-8B (dense, `Qwen/Qwen3-8B` on Tinker, $0.40/M train tokens)
- **Training platform:** Tinker (cloud LoRA training, no local GPUs needed)
- **SFT data:** Both sides of GPT-5.4 mirror matches (not just winners), since all decisions are from a strong model
- **Reward shape:** Sparse (win/loss only) for RL phases. SFT uses cross-entropy on expert play.

---

## Phase 0: Collect SFT Training Data [COMPLETED]

Collected 20 GPT-5.4 vs GPT-5.4 Cowboy mirror matches. Used both sides (not just winners) since all decisions are from a strong model.

### What we ran

```bash
scripts/rl_collect_sft.sh 20
```

### Results

- **20 matches completed, 0 failures**
- 11 P1 wins, 9 P2 wins (roughly 50/50 as expected for mirror)
- 84 avg turns/match
- **1,672 total training examples** (836 per side)
- 0 fallbacks from GPT-5.4 (perfect format compliance)
- Avg prompt: ~20K chars, avg completion: ~275 chars

### Data preparation

```bash
uv run python scripts/rl_prepare_sft.py
```

Output: `runs/rl_sft_cowboy/sft_training_data.jsonl` -- 1,672 (prompt, completion) pairs.

**Config:** `daemon/config/rl_sft_cowboy_mirror.json`
- GPT-5.4 vs GPT-5.4, Cowboy assigned, 750 starting HP, training_room stage

**Cost:** ~$30 in OpenRouter API calls for 20 matches.

---

## Baseline Eval: Qwen3-8B (No Training) [COMPLETED]

Before any training, we evaluated the off-the-shelf Qwen3-8B against Gemini and GPT-5.4 using the old (pre-hitbox-change) prompts.

### What we ran

```bash
scripts/rl_eval.sh --model "qwen/qwen3-8b" --num-matches 3 --tag "baseline_old_prompts"
```

### Results

| Opponent | Record | Avg HP Diff | Fallback Rate |
|---|---|---|---|
| Gemini 3.1 Pro | **0W-3L** | -377 | 46% |
| GPT-5.4 | **0W-3L** | -244 | 38% |

Per-game vs Gemini: 89-310, 2-459, 1-455 (destroyed every time).
Per-game vs GPT-5.4: 185-246, 103-640, 49-183 (consistently lost).

**Fallback analysis:** 79% of fallbacks were the model picking a move name not legal in the current game state (e.g., choosing "Stinger" when mid-animation with only "Continue" available). 21% were payload validation failures. The model produces valid JSON but doesn't understand state-dependent legality.

Results saved: `runs/rl_eval_baseline_old_prompts/eval_results.json`

---

## Phase 1: Supervised Fine-Tuning (SFT) [COMPLETED]

### What we ran

```bash
# Prepare training data (both sides of all 20 matches)
uv run python scripts/rl_prepare_sft.py

# Train (requires TINKER_API_KEY in .env)
set -a && source .env && set +a
PYTHONUNBUFFERED=1 uv run --project daemon python scripts/rl_save_checkpoint.py
```

### Training details

- **Model:** `Qwen/Qwen3-8B` on Tinker
- **Method:** LoRA rank 16, cross-entropy loss
- **Data:** 1,622 train + 50 eval examples (from 1,672 total)
- **Hyperparameters:** lr=0.0002, batch_size=4, 3 epochs
- **Tokenization:** `tinker_cookbook.renderers.get_renderer("qwen3")` with `TrainOnWhat.LAST_ASSISTANT_MESSAGE` (masks prompt tokens, trains only on completion)
- **Avg tokens/example:** 5,851
- **Total tokens/epoch:** ~9.5M

### Training curve

| Epoch | Train Loss | Eval Loss | Time |
|---|---|---|---|
| 1 | 0.978 | 0.758 | 19 min |
| 2 | 0.644 | 0.573 | 19 min |
| 3 | 0.475 | 0.458 | 19 min |

No overfitting (eval loss tracks train loss). Total training time: ~57 min. **Cost: ~$11.**

### Checkpoint

```
Weights: tinker://d3d1239e-cb3f-5cf5-b675-3cbb2cd928c2:train:0/weights/yomi-sft-cowboy-v1
Sampler: tinker://d3d1239e-cb3f-5cf5-b675-3cbb2cd928c2:train:0/sampler_weights/yomi-sft-cowboy-v1
```

The checkpoint is served via Tinker's OpenAI-compatible API:
- Base URL: `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`
- Model ID: the sampler path above
- Auth: `TINKER_API_KEY`

**Config:** `daemon/config/rl_eval_sft_v1.json` -- uses `provider: "openrouter"` with `base_url` option pointing at Tinker's API.

### SFT v1 Eval Results

```bash
# Ran 3 matches each vs Gemini and GPT-5.4 with daemon/config/rl_eval_sft_v1.json
```

| Opponent | Record | Avg HP Diff | Fallback Rate |
|---|---|---|---|
| Gemini 3.1 Pro | **2W-1L** | **+232** | 0% |
| GPT-5.4 | **2W-1L** | **-12** | 0% |

Per-game vs Gemini:
- [L] HP: 55 vs 215 (diff -160, 108 turns)
- [W] HP: 506 vs 99 (diff +407, 66 turns)
- [W] HP: 490 vs 42 (diff +448, 46 turns)

Per-game vs GPT-5.4:
- [W] HP: 205 vs 93 (diff +112, 100 turns)
- [W] HP: 155 vs 12 (diff +143, 70 turns)
- [L] HP: 75 vs 365 (diff -290, 138 turns)

Results saved: `runs/rl_eval_sft_v1/eval_results.json`

### Key observations

1. **Format compliance fixed:** 0% fallback rate vs 42% baseline. The SFT completely eliminated malformed/illegal output.
2. **Competitive with frontier models:** 67% win rate against both Gemini 3.1 Pro and GPT-5.4, up from 0%.
3. **Wins are decisive:** When the SFT model wins, it wins big (+407, +448, +112, +143 HP diff). The losses are closer.
4. **Only 20 matches of training data needed:** 1,672 examples from GPT-5.4 self-play was sufficient.
5. **Cheap:** ~$30 for data collection + ~$11 for training = **~$41 total**.

### Tactical analysis

The SFT didn't just fix format compliance -- it fundamentally changed how the model plays. See **[Tactical Analysis: Baseline vs SFT v1](rl_tactical_analysis.md)** for the full breakdown. Key shifts:

- **Learned to defend:** Baseline never intentionally blocked (<3% defensive). SFT uses ParryHigh (8.3%) and Roll (4.9%) as deliberate choices.
- **Learned range-specific tools:** GunThrow at long range (30% of long-range decisions), LightningSlice at mid (15%), Grab at close (10%). Baseline spammed Lasso at every range (19%).
- **Learned the RPS dynamic:** Responds to blocks with grabs, grabs with attacks, attacks with blocks. Baseline showed no counter-adaptation.
- **Learned meaningful DI:** 88% non-zero directional influence with 34+ unique values. Baseline was mostly (0,0) from the fallback handler.
- **Coverage gap:** Both models use ~28 of ~53 available moves. Moves like 3Combo, SpotDodge, and 1000Cuts are never used (inherited from GPT-5.4's preferences).

---

## Phase 2: Rejection Sampling / Expert Iteration [NOT STARTED]

Iteratively improve the model by collecting matches with the current SFT policy, keeping only winning trajectories, and SFT-ing again.

Given the strong SFT v1 results (67% win rate), this phase may not be necessary. Consider going directly to GRPO if the goal is to push win rate above 80%.

### Loop

```
for each iteration:
    1. Run matches with current model vs Gemini/GPT-5.4
    2. Filter to winning trajectories
    3. SFT on original expert data + new winning trajectories
    4. Evaluate
    5. If win rate improves, keep; otherwise stop
```

---

## Phase 3: GRPO [NOT STARTED]

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
    trajectories = await collect_rollouts(sc, num_matches=50)

    # 3. Group by match, compute advantages
    groups = group_by_match(trajectories)
    advantages = compute_group_relative_advantages(groups)

    # 4. Build training data and train
    data = assemble_training_data(groups, advantages)
    await tc.forward_backward_async(data, loss_fn="ppo")
    await tc.optim_step_async(AdamParams(learning_rate=5e-5))

    # 5. Evaluate every N iterations
    if iteration % 10 == 0:
        evaluate(sc)
```

### Self-play (optional extension)

1. Freeze current best model as opponent
2. Train against frozen opponent
3. Periodically update frozen opponent with latest weights
4. Maintain a league of past checkpoints to avoid strategy cycling

---

## Evaluation

### Quick eval (during development)

```bash
# Use the SFT config directly with run_match.sh
scripts/run_match.sh --daemon-config daemon/config/rl_eval_sft_v1.json \
    --runs-root runs/rl_eval_sft_v1 --no-replay --skip-mod-push
```

### Batch eval

```bash
# For models served via OpenRouter (baseline, off-the-shelf models)
scripts/rl_eval.sh --model "qwen/qwen3-8b" --num-matches 3 --tag "baseline"

# For Tinker-hosted models, use the dedicated config
# (daemon/config/rl_eval_sft_v1.json already configured)
```

### Reading results

Results are serialized to `runs/rl_eval_<tag>/eval_results.json`:

```json
{
  "model": "tinker/yomi-sft-cowboy-v1",
  "opponents": {
    "google/gemini-3.1-pro-preview": {
      "games": 3, "wins": 2, "losses": 1,
      "avg_hp_diff": 232.0, "p1_fallback_rate": 0.0
    }
  },
  "games": [...]
}
```

Key metrics:
- **Win rate** vs each opponent
- **Avg HP differential** (positive = our model ahead at match end)
- **Fallback rate** (malformed/illegal output rate -- should be 0% after SFT)
- **Avg turns** (very short = decisive wins; very long = passive stalls)

---

## Whiff Rate Analysis

All models whiff 55-67% of attacks regardless of prompt version. This is inherent to the simultaneous-move game (both players move at once, so position predictions are uncertain).

| Model | Old Prompts Whiff% | New Prompts Whiff% |
|---|---|---|
| Gemini 3.1 Pro | 63.7% | 55.3% |
| GPT-5.4 | 67.1% | N/A (small sample) |
| Qwen3-8B | 61.4% | 56.9% |

**Note on the hitbox data update:** We expanded hitbox coverage significantly (projectile parsing, state-level data, programmatic overrides) but the updated `v_range` data made the range guidance too strict for airborne situations, causing models to play passively. The prompt code was reverted to `ca04967` for now. The extraction improvements are preserved in `scripts/extract_hitbox_data.py` for future use once the range guidance handles vertical distances better.

---

## Known Issues

1. **Range guidance vertical check too strict:** The `_range_guidance` function says "NO attacks would hit" 36% of turns (almost all when someone is airborne). This makes models overly passive. Needs fixing before re-enabling the expanded hitbox data.

2. **Long matches:** At 750 HP, matches average 84-135 turns. With ~30s per turn, a single match takes 5-15 minutes. Serial eval of 6 matches takes 1-2 hours.

3. **Game stability:** The Godot game occasionally disconnects mid-match ("Connection lost: no close frame received"). Matches must be retried. The eval scripts handle this gracefully.

---

## Clock Time Budget

Actual measured times from our experiments:

| Task | Matches | Time (serial) | Cost |
|---|---|---|---|
| Phase 0: SFT data collection | 20 | ~2.5 hours | ~$30 |
| Phase 1: SFT training | -- | ~57 min | ~$11 |
| Baseline eval (3+3 matches) | 6 | ~2 hours | ~$5 |
| SFT eval (3+3 matches) | 6 | ~1.5 hours | ~$5 |

Mitigations for slow clock time:
1. **Parallel Godot instances** (each on a different port) -- not yet implemented
2. **`Engine.time_scale`** acceleration in Godot (untested)
3. **`--no-replay` and `--skip-mod-push`** flags (already used, saves ~30s per match)
4. **750 HP** instead of 1500 (already configured)

---

## File Reference

| File | Purpose |
|---|---|
| `scripts/rl_collect_sft.sh` | Run N SFT data collection matches |
| `scripts/rl_prepare_sft.py` | Convert match decisions to training JSONL |
| `scripts/rl_train_sft.py` | SFT training on Tinker (with eval and sanity check) |
| `scripts/rl_save_checkpoint.py` | Train and save persistent Tinker checkpoint |
| `scripts/rl_eval.sh` | Evaluate model via OpenRouter, serialize results |
| `scripts/extract_hitbox_data.py` | Extract hitbox data from decompiled .tscn files |
| `daemon/config/rl_sft_cowboy_mirror.json` | GPT-5.4 mirror config for SFT data |
| `daemon/config/rl_eval_vs_gemini.json` | Generic eval config (patches model ID) |
| `daemon/config/rl_eval_sft_v1.json` | Eval config for SFT v1 via Tinker API |
| `daemon/src/yomi_daemon/rl/rewards.py` | Composable reward functions |
| `daemon/src/yomi_daemon/rl/trajectory.py` | Step, Trajectory, MatchOutcome types |
| `daemon/src/yomi_daemon/rl/collector.py` | TrajectoryCollector for live matches |
| `daemon/src/yomi_daemon/rl/tinker_env.py` | Prompt/completion formatting for Tinker |
| `daemon/src/yomi_daemon/rl/serialization.py` | JSONL read/write for trajectories |
| `daemon/src/yomi_daemon/rl/features.py` | Numeric feature extraction from observations |
| `docs/rl_brainstorming.md` | Extended brainstorming and analysis |
| `docs/rl_libraries_infra.md` | Tinker and OpenReward research |
| `docs/rl_decisions.md` | Key experiment decisions and rationale |
| `runs/rl_sft_cowboy/` | SFT training data (20 matches) |
| `runs/rl_eval_baseline_old_prompts/` | Baseline eval results (6 matches) |
| `runs/rl_eval_sft_v1/` | SFT v1 eval results (6 matches) |
