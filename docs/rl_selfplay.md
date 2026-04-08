# RL Self-Play Plan

After SFT v1 reached 85%/70% win rates vs Gemini/GPT-5.4, the next step is RL self-play to push beyond what distillation can achieve. This doc covers the planned runs, evaluation methodology, and pass/fail criteria.

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

### Evaluation after run 1

Run the RL iter 5 checkpoint through the same eval used for SFT v1:
- 20 matches vs Gemini 3.1 Pro
- 20 matches vs GPT-5.4
- **10 matches vs frozen SFT v1 (most important comparison)**

**Cost:** ~50 matches × 7 min ≈ 5-6 hours, ~$25 in API calls.

The critical comparison is **RL iter 5 vs frozen SFT v1**:

| Outcome | Interpretation |
|---|---|
| RL > SFT (60%+ win rate) | Self-play discovered better strategy. Loop works. |
| RL ≈ SFT (45-55% win rate) | No change. Either too little training signal or already at ceiling. |
| RL < SFT (35-45%) | Self-play introduced noise or regressed. Bad lr, bad reward, or bug. |
| RL << SFT (<30%) | Policy degraded badly. Reduce lr or abort. |

### Tactical analysis on RL iter 5 matches

After eval, run the same tactical breakdown we did for SFT v1:
- Did the opening move distribution change?
- Is it using moves the SFT never used (3Combo, SpotDodge, VSlash)?
- Did DI usage evolve?
- Did action diversity increase or decrease?

If there are tactical shifts even without win-rate improvement, that's still informative -- it tells us whether self-play is exploring or just adding noise.

---

## Run 2: Real training run (if run 1 passes)

Assuming run 1 satisfies the pass criteria, run 2 is the actual training run.

### Config

- **15-20 iterations**
- **8 matches per iteration** = 120-160 total matches
- Same lr, batch size, reward shape as run 1
- Eval every 5 iterations against frozen SFT v1 (small 5-match eval, cheap)
- Final eval: 20 matches vs each of Gemini, GPT-5.4, SFT v1

### Estimated cost and time

| Item | Value |
|---|---|
| Rollout matches | ~150 × ~7 min ≈ 17 hours |
| Training cost | ~$30 |
| Sampling cost | ~$30 |
| Eval cost | ~$25 |
| **Total** | **~$85, ~20 hours clock time** |

This is the run we'd expect to actually produce a stronger model, if self-play is going to work at all. By iter 20, the model will have seen ~160 matches of self-play data on top of the original 1672 SFT examples.

---

## Run 3: HP delta shaping (only if run 2 stalls)

If sparse self-play converges but plateaus before beating SFT v1, try adding dense rewards:

- Same config as run 2
- `--hp-delta-weight 0.05` (small shaping term)
- Watch for reward hacking: avg HP margin rising faster than win rate

If HP delta shaping helps, push the weight up. If it causes ultra-defensive drift (avg turns per match increases, attack rate drops), back off.

**Why we don't start with dense rewards:** the SFT model already learned HP-efficient play from GPT-5.4 traces (avg 429 HP remaining in wins). We don't need dense signal for early progress, and we want clean differentiation between sparse RL and SFT for the first comparison. Dense rewards introduce reward hacking risk that's harder to detect than slow convergence.

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

**Decision:** start sparse for run 1 and run 2. The SFT starting point is strong enough that we don't need the variance reduction. If sparse stalls, run 3 adds shaping at low weight with explicit reward-hacking monitoring.

---

## Open questions

These don't block the first runs but should be addressed eventually:

1. **Does self-play converge to a Nash equilibrium or to a single dominant strategy?** Mirror self-play in a fighting game can either find balanced play or collapse to one side dominating. We're not sure which until we run it.

2. **How exploitable is the converged policy?** Even if RL produces a strong-vs-frontier-models checkpoint, an adversary trained specifically to counter it (anti-RL training) would expose its weaknesses. Worth doing as a separate experiment.

3. **Does the model discover moves the SFT never used?** SFT v1 uses 28 of ~53 legal moves. If self-play unlocks 3Combo, SpotDodge, 1000Cuts, etc., that's qualitative evidence of exploration.

4. **What's the right exploration temperature?** Run 1 uses 0.9 in rollouts. If the model is too deterministic, it won't explore. If too random, it won't execute combos. Worth tuning if run 1 shows mode collapse or insufficient action diversity.

5. **Can we parallelize rollouts?** Each match takes ~7 minutes serially. With 8 parallel game instances on different ports, we'd cut iteration time 8x. Not implemented yet -- this is the biggest single throughput improvement available.
