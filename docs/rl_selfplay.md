# RL Self-Play Plan

After SFT v1 reached 85%/70% win rates vs Gemini/GPT-5.4, the next step is RL self-play to push beyond what distillation can achieve. This doc covers the planned runs, evaluation methodology, and pass/fail criteria.

> **Status note:** This document records the original/self-play-v1 plan and historical run results. For the current diagnosis of why RL has struggled to beat SFT reliably, plus the proposed fix set for the next iteration of training, see [RL v2 Plan](rl_v2.md).

## Background

The self-play loop is implemented in `scripts/rl_self_play.py`. It uses REINFORCE with a batch baseline (mathematically equivalent to Tinker's `cross_entropy` loss with advantage-scaled token weights). Per iteration:

1. Save current weights as a Tinker sampler checkpoint
2. Run N self-play matches via the daemon (both P1 and P2 use the current policy)
3. Parse trajectories from `runs/`, compute per-step rewards
4. Compute advantages with batch-mean baseline + z-score normalization
5. Tokenize each (prompt, action) with advantage-scaled weights
6. Forward/backward with `cross_entropy`
7. Save persistent checkpoints every N iterations

The starting point is the SFT v1 checkpoint (`tinker://d3d1239e-cb3f-5cf5-b675-3cbb2cd928c2:train:0/weights/yomi-sft-cowboy-v1`).

---

## Run 1: Feasibility check (small, cheap)

The goal of run 1 is **not** to get SOTA results -- it's to answer: "does our self-play loop work at all, and is it finding new behavior?"

### Config

- Starting point: SFT v1 checkpoint
- **5 iterations**
- **4 matches per iteration** = 20 total self-play matches
- Sparse reward (win/loss only, no HP delta shaping)
- `lr = 5e-5` (10x lower than SFT lr; we're fine-tuning a trained model)
- `batch_size = 4`
- Temperature 0.9 in rollouts (the template config)
- Persistent checkpoint saved after iter 5

### Estimated cost and time

| Item | Value |
|---|---|
| Rollout matches | 20 × ~7 min = ~2.5 hours |
| Training tokens | ~20 matches × ~100 steps × ~5800 tok ≈ 11.6M tokens |
| Training cost | 11.6M × $0.40/M ≈ $5 |
| Sampling cost | 11.6M × $0.40/M ≈ $5 |
| **Total** | **~$10, ~3 hours** |

### Command

```bash
set -a && source .env && set +a
PYTHONUNBUFFERED=1 uv run --project daemon python scripts/rl_self_play.py \
    --num-iterations 5 --matches-per-iter 4
```

### What we're looking for

This run is a sanity check, not a results run. Watch for:

1. **Does training loss decrease?** The cross-entropy loss weighted by advantage should show a negative trend if the policy is actually updating. Flat loss = nothing is happening.

2. **Does action distribution shift?** Compare the SFT v1 action frequencies to iter 5. If they're identical, self-play isn't changing the policy. If they drift, something is happening (good or bad).

3. **Does the opening move change?** SFT v1 opens HSlash2 95% of the time. If self-play finds HSlash2 being exploited (both players open it, both take damage), we'd expect the opening to diversify -- maybe more ParryHigh or Lasso openings.

4. **Do match outcomes stay balanced?** In self-play mirror, P1 and P2 are the same model, so matches should stay near 50-50. If one side starts winning >70%, the model is developing asymmetric strategies (probably overfitting to state features that shouldn't matter).

5. **Does fallback rate stay at 0%?** Policy gradient can degrade output format if the advantage signal pushes the model off-distribution. If fallback rate climbs above 5%, we're destroying the format compliance that SFT achieved.

### Pass criteria for run 1

- ✅ Training loss trending downward
- ✅ Fallback rate stays ≤5%
- ✅ Mirror match outcomes stay 40-60% for either side (no collapse)
- ✅ Action distribution shows some drift from SFT v1 (not completely static)

### Fail criteria (abort and redesign)

- ❌ Fallback rate climbs above 20% (format destroyed)
- ❌ Mirror matches go 0-20 or 20-0 (policy collapsed asymmetrically)
- ❌ Model starts producing the same action 90%+ of the time (mode collapse)
- ❌ Training loss diverges (NaN, exploding)

### Run 1 results (2026-04-08 to 2026-04-10)

We ran multiple iterations of run 1 to debug and stabilize the training loop.

**Run 1a (original config: sparse, lr=5e-5, no anchor):**
- FAILED. Model collapsed by iter 4 — all matches timing out at 900s, fallback rate >20%.
- Root cause: unbounded advantage scaling (±1.0 after z-norm) with lr=5e-5 pushed log-probs of losing actions toward -∞. No KL anchor meant drift compounded across iterations.
- Loss spiked to 9.28 in a single batch at iter 3. By iter 4, model couldn't produce valid JSON.

**Fixes applied (run 1c):**
- `lr`: 5e-5 → 5e-6 (10x reduction)
- `adv-clip`: unbounded → 0.2 (caps gradient magnitude)
- `save-every`: 5 → 1 (checkpoint every iter for inspection)
- Match timeout: 900s → 600s
- Fallback-rate guard: abort if >20% in any iter
- Namespaced run dirs to prevent stale-data contamination

**Run 1c (conservative sparse):**
- All 5 iters completed. Loss bounded, fallback ≤0.6%.
- Mirror: aggregate P1 14-6 (P1 bias emerging).
- Eval vs SFT v1: **RL 4-6 (40% win rate)**. No improvement over SFT.

**Run 2a (added SFT replay anchor at 0.5 ratio):**
- All 5 iters completed. P1 bias eliminated: aggregate P1 10-10.
- SFT anchor loss decreased monotonically (0.3416 → 0.2478), confirming policy stayed near SFT.
- Eval vs SFT v1: **RL 4-6 (40% win rate)**. Anchor fixed stability but still no signal.

**Run 3b (added HP-delta shaping at weight 0.05):**
- All 5 iters completed. Fallback ≤0.6%, loss bounded.
- Mirror: aggregate P1 12-8 (mild P1 lean, within noise).
- Eval vs SFT v1 (RL as P1): **RL 6-4 (60%)**
- Eval vs SFT v1 (RL as P2): **RL 7-3 (70%)**
- **Combined: RL 13-7 (65% win rate) across 20 matches.** First positive result.
- HP margins on RL wins averaged +338, vs -186 on losses — RL wins are more decisive.

**Key takeaways:**
1. Sparse reward with N=4 matches is below the noise floor for self-play. HP-delta shaping was essential.
2. SFT replay anchor prevents drift and P1 bias but doesn't add learning signal alone.
3. Conservative lr (5e-6) + advantage clipping (0.2) prevents the catastrophic collapse seen in 1a.
4. The working hyperparameter combination: `lr=5e-6, adv-clip=0.2, hp-delta-weight=0.05, sft-replay-ratio=0.5`.

---

## Run 2: Real training run (run 1 passed)

Run 1 confirmed that HP-delta shaping + SFT anchor + conservative lr produces genuine improvement (65% vs SFT v1). Run 2 scales up to get a stronger checkpoint and test whether improvement is monotonic.

### Config

- Starting point: SFT v1 checkpoint (fresh start, not resuming from run 1)
- **15 iterations**
- **8 matches per iteration** = 120 total matches
- `lr = 5e-6`
- `adv-clip = 0.2`
- `hp-delta-weight = 0.05`
- `sft-replay-ratio = 0.5` (KL-style anchor)
- `batch_size = 4`
- Temperature 0.9 in rollouts
- Persistent checkpoint saved every iteration

### Command

```bash
set -a && source .env && set +a
PYTHONUNBUFFERED=1 uv run --project daemon python scripts/rl_self_play.py \
    --num-iterations 15 --matches-per-iter 8 \
    --lr 5e-6 --adv-clip 0.2 --hp-delta-weight 0.05 --sft-replay-ratio 0.5
```

### Estimated cost and time

| Item | Value |
|---|---|
| Rollout matches | 120 × ~5 min ≈ 10 hours |
| Training (RL + SFT replay) | ~$40 |
| Sampling cost | ~$40 |
| **Total** | **~$80, ~12-15 hours clock time** |

### Pass criteria

- RL iter 15 beats SFT v1 >65% (across both P1 and P2 sides)
- RL iter 15 competitive with GPT-5.4 (>40% win rate)
- Monotonic improvement: iter 15 > iter 10 > iter 5 > SFT v1
- No fallback rate spike or mirror collapse at any point

### Evaluation plan

After training, round-robin eval:
- Checkpoints iter 5, 10, 15 each vs frozen SFT v1 (10 matches, both sides)
- Iter 15 vs GPT-5.4 (10 matches, both sides)
- Build win-rate curve to confirm monotonic improvement

### Run 2b results (2026-04-13 to 2026-04-14)

We ran a variant of Run 2 with **lower HP-delta shaping**:
- `hp-delta-weight = 0.01` (vs 0.05 in run 3b)
- Other hyperparameters unchanged: `lr=5e-6, adv-clip=0.2, sft-replay-ratio=0.5`

Because the run was resumed across multiple shells after crashes, the final artifact naming is a little messy: the final persistent checkpoint is `tinker://61069a9a-8c75-557d-97ee-29d530f5bdb6:train:0/weights/rl-selfplay-iter11`, but it corresponds to **overall run2b iter 15**.

**Training run summary:**
- Effective training iterations completed: **15/15**
- Planned matches: **120**
- Matches successfully parsed into training: **114**
- Dropped matches: **6** total
  - **5 disconnects** (`Connection lost during match: no close frame received or sent`)
  - **1 timeout race** where `run_match.sh` marked the match failed at 600s, but the daemon finalized it moments later, after trajectory parsing had already begun
- Mirror aggregate across parsed matches: **P1 56W / P2 58W** (balanced overall)
- Iteration fallback rates stayed low: **0.0% to 0.5%**

**Eval vs frozen SFT v1 (20 matches total, both sides):**
- RL as **P1**: **5-5**, avg HP diff **-10**
- RL as **P2**: **7-3**, avg HP diff **+263**
- **Combined: RL 12-8 (60% win rate)**
- Combined avg HP diff: **+127**
- Avg turns per game: **79.4**
- RL fallback rate: **2 / 794 decisions = 0.25%**
- SFT fallback rate: **1 / 794 decisions = 0.13%**

**Interpretation:**
1. Lowering HP-delta shaping from **0.05 → 0.01** still produced a model that beats SFT v1, but it **underperformed** the earlier run 3b result (**60% vs 65%**).
2. The lower shaping weight did **not** buy meaningfully better format stability; fallback rates were already low at 0.05 and remained low here.
3. The run does **not** meet the original Run 2 pass criterion of **>65% vs SFT v1**.
4. Current evidence favors keeping **`hp-delta-weight = 0.05`** as the better default. If we continue scaling Run 2-style training, the next serious long run should use **0.05**, not **0.01**.

---

## Run 3: HP delta weight tuning (only if run 2 plateaus)

Run 1 showed that HP-delta weight 0.05 is sufficient to produce signal at N=4. At N=8 (run 2), we may find that 0.05 is either too weak (plateau early) or too strong (reward hacking). If run 2 plateaus:

- Try `--hp-delta-weight 0.10` or `0.02`
- Watch for reward hacking: avg HP margin rising faster than win rate
- Watch for ultra-defensive drift: avg turns per match increases, attack rate drops
- Compare iter-by-iter win rate curves across different weights

---

## Run 4: SFT vs RL head-to-head tournament

Once we have multiple RL checkpoints (iter 5, iter 10, iter 15, iter 20), run a round-robin tournament:

- Each checkpoint plays every other checkpoint (including frozen SFT v1)
- 5 matches per pairing
- Build an Elo ladder

This tells us if training is monotonically improving. If iter 20 beats iter 15 beats iter 10 beats SFT, we have a clean training curve. If the ranking is messy, training is adding variance rather than skill, and we need to debug.

---

## Why REINFORCE instead of GRPO/PPO

GRPO needs multiple completions from the same prompt to compute group-relative advantages. That doesn't fit a multi-turn game where each state is seen exactly once per match.

PPO needs old-policy logprobs we'd have to track through the rollout pipeline. Doable but adds infrastructure.

REINFORCE with batch baseline is simpler, works with Tinker's `cross_entropy` loss directly via advantage-scaled token weights, and is sufficient given we're starting from a strong SFT prior. The math:

```
loss = -sum(weight * logprob)              # standard cross-entropy
weight = base_mask * advantage              # advantage scaling
loss = -sum(advantage * base_mask * logprob)
     = -advantage * sum(logprob over completion)
```

Minimizing this maximizes `advantage * log p(action)`, which is the REINFORCE objective. Positive advantage → increase log-prob of action. Negative advantage → decrease log-prob.

If REINFORCE proves insufficient, the upgrade path is:
1. Add HP delta shaping (denser signal)
2. Switch to importance sampling loss with old logprobs (variance reduction)
3. Switch to PPO clipped loss (more stable for off-policy)

## Why sparse rewards instead of dense

**Pros of dense (HP delta shaping):**
- 50x more reward events per iteration → faster convergence
- Better credit assignment within a match
- Lower variance signal
- Distinguishes "both sides improving" from "stagnant" in self-play

**Cons of dense:**
- Reward hacking: model optimizes the proxy instead of winning
- HP delta isn't a perfect proxy for win probability
- Penalizes setup plays that take damage to create bigger advantages
- Magnitude calibration is fiddly (terminal vs dense balance)
- Hard to detect when shaping has gone wrong

**Decision (updated after run 1 results):** HP-delta shaping at weight 0.05 is now the default. Run 1 showed that sparse-only reward produces no learning signal at N=4 matches/iter (40% vs SFT after 5 iters). Adding HP-delta at 0.05 moved the needle to 65%. The SFT replay anchor (ratio 0.5) mitigates reward hacking risk by keeping the policy close to SFT v1's data distribution.

---

## Open questions

These don't block the first runs but should be addressed eventually:

1. **Does self-play converge to a Nash equilibrium or to a single dominant strategy?** Mirror self-play in a fighting game can either find balanced play or collapse to one side dominating. We're not sure which until we run it.

2. **How exploitable is the converged policy?** Even if RL produces a strong-vs-frontier-models checkpoint, an adversary trained specifically to counter it (anti-RL training) would expose its weaknesses. Worth doing as a separate experiment.

3. **Does the model discover moves the SFT never used?** SFT v1 uses 28 of ~53 legal moves. If self-play unlocks 3Combo, SpotDodge, 1000Cuts, etc., that's qualitative evidence of exploration.

4. **What's the right exploration temperature?** Run 1 uses 0.9 in rollouts. If the model is too deterministic, it won't explore. If too random, it won't execute combos. Worth tuning if run 1 shows mode collapse or insufficient action diversity.

5. **Can we parallelize rollouts?** Each match takes ~7 minutes serially. With 8 parallel game instances on different ports, we'd cut iteration time 8x. Not implemented yet -- this is the biggest single throughput improvement available.
