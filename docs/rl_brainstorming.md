# RL for YOMI Hustle: Brainstorming Document

## The Core Question

Can we RL-train a small LLM to consistently beat Gemini 3.1 Pro at YOMI Hustle?

The existing system already frames the game as a text-based decision problem: structured game state in, named action out. This is not a pixel-observation RL problem - it's an LLM reasoning problem with a well-defined action space. That makes LLM post-training (GRPO/PPO on text) the natural approach.

---

## Current Architecture (What We Already Have)

The existing daemon + mod bridge already provides everything an RL environment needs:

| RL Concept | Existing Implementation |
|---|---|
| **Observation** | `DecisionRequest` - structured game state (HP, position, velocity, meter, burst, state, history) |
| **Action space** | `LegalActions` - variable-size discrete set of named moves with optional parameters |
| **Dense reward** | `HistoryEntry.p1_hp_delta / p2_hp_delta` - per-turn HP change |
| **Terminal reward** | `MatchEnded.winner` - win/loss signal |
| **Policy interface** | `PolicyAdapter.decide(request) -> ActionDecision` - exactly what an RL policy implements |
| **Rollout collection** | `decisions.jsonl` - full trajectory logging already exists |
| **Evaluation** | Tournament system with Elo ratings |

**Key insight:** We don't need to build a gym env from scratch. We need a training loop wrapper around the existing system.

---

## Single-Turn Architecture: A Feature, Not a Bug

Our current architecture treats each decision as an independent prompt with selected context from previous rounds (history entries with HP deltas, outcomes, positions). This is **ideal for GRPO training:**

- **Direct GRPO mapping.** GRPO is naturally single-turn: generate N completions for the same prompt, score them, compute group-relative advantages. Each game state is one "problem" with one "answer." We use Tinker's `ProblemEnv`, not `MessageEnv`.
- **No long-horizon credit assignment.** In multi-turn RL, you must attribute a win/loss back through 30+ turns (hard, noisy). With independent prompts + per-turn HP delta rewards, credit assignment is immediate: "this action in this state caused this damage."
- **Context window efficiency.** Each training example is one game state + action, not an entire match transcript. Cheaper per-token, more rollouts per dollar.
- **No vanishing gradients over time.** Multi-turn RLHF struggles with gradient signal fading over long sequences. Single-turn avoids this entirely.

**What matters:** The quality of the selected context (history entries). For RL to work well, the model needs to see the consequences of past actions so it can learn patterns like "I used X last turn and got punished, so Y is better now." The existing `HistoryEntry` includes `p1_hp_delta`, `p2_hp_delta`, `p1_outcome`, `p2_outcome`, actions chosen, and positions - this is sufficient for learning reactive strategy. Worth auditing that the most decision-relevant signals (opponent pattern recognition, outcome of last interaction) are reliably populated.

---

## Model Size Analysis: What's the Smallest Model That Could Work?

### Why Model Size Matters

- **Inference latency:** Each turn has a deadline (~10s for LLM matches). Smaller models respond faster, enabling more rollouts per hour.
- **Training cost:** Tinker charges per-token. Smaller models = cheaper training.
- **Throughput:** More parallel rollouts = more training data per iteration.

### What the Task Requires

The LLM must:
1. **Parse structured game state** - positions, HP, meter, move states, history
2. **Understand legal actions** - variable set of ~10-30 moves per turn with metadata
3. **Reason about strategy** - RPS dynamics (attack > grab > block > attack), spacing, resource management
4. **Select and parameterize actions** - output valid JSON with action name + optional DI, feint, prediction

### Model Size Tiers

| Size | Examples | Baseline Capability | RL Potential |
|---|---|---|---|
| **1-3B** | Qwen 2.5 1.5B/3B, Llama 3.2 1B/3B, Phi-3.5 Mini | Can follow simple JSON schemas. Weak at multi-step reasoning. | Could learn pattern matching (if X then Y) but unlikely to develop deep strategy. Might beat scripted baselines but probably not a strong LLM. |
| **7-8B** | Qwen 2.5 7B, Llama 3.1 8B, Mistral 7B v0.3, Gemma 2 9B | Solid JSON parsing, basic reasoning, can follow complex prompts. | **Sweet spot candidate.** Enough capacity to learn game-specific reasoning through RL. Small enough for high rollout throughput. |
| **13-14B** | Qwen 2.5 14B, Llama 3.1 13B (if available) | Stronger reasoning, better at nuanced decisions. | Good capacity but 2x cost of 7B. Worth it only if 7B plateau is too low. |
| **32-70B** | Qwen 2.5 32B/72B, Llama 3.1 70B | Near frontier reasoning. | Expensive but might be necessary if the task requires deep strategic reasoning that smaller models can't learn. |

### Recommendation: Start with 7-8B

**Qwen 2.5 7B** or **Llama 3.1 8B** as the primary candidate:
- Large enough to handle structured game state parsing and JSON output
- Small enough for high training throughput on Tinker
- Strong RL training results in the literature (DeepSeek-R1 started from Qwen 2.5 base models)
- If 7B plateaus, scale to 14B or 32B with the same training pipeline

**Fallback plan:** If 7B can't learn adequate strategy, try:
1. Qwen 2.5 14B (2x cost, meaningful capability jump)
2. Qwen 2.5 32B MoE (priced by active params on Tinker, potentially cost-effective)

---

## Reward Shaping

### Reward Signals Available

#### Terminal (Sparse)
- **Win/Loss:** +1 for win, -1 for loss, 0 for draw
- **KO vs Timeout:** Could weight KO wins higher (demonstrates dominance)

#### Per-Turn (Dense)
- **HP Delta:** `my_hp_delta - opponent_hp_delta` (damage dealt minus damage received)
- **Combo Extension:** Bonus for increasing combo count
- **Blockstun Inflicted:** Reward for putting opponent in blockstun (pressure)
- **Meter Gain:** Reward for building meter through successful actions

#### Positional (Shaping)
- **Center Stage Control:** Reward for occupying center, penalty for being cornered
- **Range Advantage:** Reward for being at your character's optimal range
- **Initiative:** Reward for having frame advantage / being in advantageous state

#### Efficiency
- **Fallback Penalty:** Negative reward when decision times out and falls back
- **Action Diversity:** Penalty for repeating the same action (anti-spam, already tracked as repetition warning)
- **Resource Management:** Penalty for wasting burst/meter on low-value situations

### Reward Design Principles

1. **Start sparse, add density carefully.** Win/loss alone provides the cleanest signal. Dense rewards risk reward hacking (e.g., agent learns to farm HP delta without trying to win).

2. **HP delta is the most natural dense reward.** It's directly tied to the game objective and already tracked per-turn.

3. **Avoid rewarding intermediate states too strongly.** Positional rewards (center control, range) are useful for exploration but can distort strategy if weighted too heavily.

4. **Curriculum progression:**
   - Phase 1: Win/loss only vs scripted baselines (learn basic competence)
   - Phase 2: Win/loss + HP delta vs scripted baselines (learn to deal damage efficiently)
   - Phase 3: Win/loss + HP delta vs self-play (learn to adapt to intelligent opponents)
   - Phase 4: Win/loss + HP delta vs frozen snapshots of Gemini 3.1 Pro (learn to exploit specific opponent)

### Proposed Reward Function

```python
def compute_reward(trajectory, gamma=0.99):
    """
    Composite reward for a single turn.
    """
    rewards = []
    for turn in trajectory.turns:
        r = 0.0
        
        # Dense: HP delta (normalized to [-1, 1] range)
        hp_delta = (turn.opponent_hp_delta - turn.my_hp_delta) / max_hp
        r += 0.3 * hp_delta
        
        # Penalty for fallback
        if turn.was_fallback:
            r -= 0.1
        
        rewards.append(r)
    
    # Terminal: win/loss
    if trajectory.winner == my_player:
        rewards[-1] += 1.0
    elif trajectory.winner is not None:
        rewards[-1] -= 1.0
    
    return rewards
```

### Reward Hacking Risks

| Risk | Mitigation |
|---|---|
| Agent spams one high-damage move | Repetition penalty + opponent adapts in self-play |
| Agent plays ultra-defensively to avoid negative HP delta | Win/loss terminal reward dominates |
| Agent exploits scripted baseline patterns | Graduate to self-play and frozen opponent training |
| Agent learns to time out intentionally | Fallback penalty + track as metric |

---

## Training Pipeline Design

### Phase 1: Supervised Warmup (Optional but Recommended)

Before RL, SFT the base model on expert trajectories:
- Collect games from strong LLM players (Gemini 3.1 Pro, Claude Opus)
- Filter to winning games only
- Fine-tune on (observation, action) pairs
- This gives the model a reasonable starting policy to improve from

**Why:** RL from a random policy in a complex game converges very slowly. Starting from a policy that already understands the action format and basic strategy dramatically speeds up training.

### Phase 2: GRPO Training Loop

```
for each training iteration:
    1. Save current model weights
    2. Create SamplingClient from saved weights
    3. For each batch of prompts (game situations):
        a. Generate group_size rollouts (play group_size games)
        b. Score each rollout (reward function)
        c. Compute group-relative advantages (GRPO)
        d. Skip degenerate groups
    4. Assemble training data from scored rollouts
    5. forward_backward + optim_step on Tinker
    6. Log metrics to W&B
    7. Every N iterations: evaluate against benchmark opponents
```

### Phase 3: Self-Play

Once the model beats scripted baselines consistently:
1. Freeze current best model as "opponent"
2. Train against frozen opponent
3. Periodically update frozen opponent with latest weights
4. Maintain a league of past checkpoints (avoid forgetting how to beat earlier strategies)

### Phase 4: Targeted Training

Once the model is competitive:
1. Collect Gemini 3.1 Pro games (it's the target to beat)
2. Train against a frozen Gemini 3.1 Pro policy
3. Analyze failure modes and add targeted training scenarios

---

## Environment Modifications Needed

### For Rollout Collection (Training Data)

The current system runs one game at a time. For RL training, we need:

1. **Headless game execution:** Run Godot without rendering for faster rollouts. Check if YOMI Hustle supports `--headless` mode or if we can use `xvfb-run`.

2. **Parallel game instances:** Run multiple game+daemon pairs simultaneously. Each on a different port.

3. **Episode reset:** Automatically start new matches after each game ends. The mod already handles this for tournament mode.

4. **Trajectory serialization:** Extend `decisions.jsonl` to include rewards and advantages, or create a separate trajectory format for Tinker.

### For the Training Loop

1. **Tinker environment wrapper:** Implement `tinker_cookbook.rl.Env` that:
   - `initial_observation()` returns the first turn's game state as text
   - `step(action)` sends the action to the game, waits for next turn, returns (reward, done, metrics)

2. **Batch collection:** Collect N complete episodes, compute rewards, feed to Tinker.

3. **Prompt format for RL:** Likely use a simplified version of the current prompt templates. Less prose, more structured state. The model learns what matters through RL rather than being told.

### For Evaluation

1. **Benchmark suite:** Fixed set of opponents (random, block_always, greedy_damage, scripted_safe, frozen past checkpoints, Gemini 3.1 Pro)
2. **Elo tracking:** Already exists in tournament system
3. **Win rate over N games:** Statistical significance requires ~100+ games per matchup

---

## Challenges and Open Questions

### 1. Rollout Speed

**Problem:** Each game takes real time in Godot (not accelerated). At ~30 turns per match, ~1s per turn = ~30s per game. To generate 1000 rollouts per training iteration, that's ~8 hours per iteration.

**Mitigations:**
- Headless mode (skip rendering) - may get 5-10x speedup
- Parallel game instances - linear speedup with # instances
- Shorter matches (could train on first-to-X-damage rather than full matches)
- Game speed modification via Godot Engine.time_scale

**Open question:** Can we set Godot's `Engine.time_scale` to 10x or higher for training? Does the game logic break at high speeds?

### 2. Variable Action Space

**Problem:** Legal actions change every turn (different moves available based on state). Standard RL assumes fixed action spaces.

**Solution:** This is handled naturally in the LLM setting. The legal actions are part of the prompt text. The model learns to select from whatever actions are available. GRPO handles this by generating multiple completions and scoring them.

### 3. Simultaneous Actions

**Problem:** Both players act simultaneously. This is not a standard MDP - it's closer to a simultaneous-move game (like rock-paper-scissors each turn).

**Implications:**
- No "opponent's turn" to observe before acting
- Prediction of opponent's action is valuable (already supported via `prediction` field)
- Self-play training naturally handles this (both sides are the same policy)
- Nash equilibrium concepts apply (mixed strategies may be optimal)

### 4. Character Matchups

**Problem:** 5 characters x 5 opponents = 25 matchups. Each plays differently.

**Options:**
- Train one universal policy (harder but more general)
- Train per-character specialists (5 models, each expert at one character)
- Train per-matchup specialists (25 models, overkill)

**Recommendation:** Start with one character (probably Ninja or Robot based on data availability) against all opponents. Generalize later.

### 5. Observation Size

**Problem:** Full game state serialized as text can be 2000+ tokens. With 10-turn history, that's substantial context per turn. Training cost scales with tokens.

**Mitigations:**
- Compact observation format (minimize tokens without losing information)
- Truncate history to last 3-5 turns
- Use structured/tabular format rather than prose
- Consider whether the model needs all fields or a curated subset

### 6. Stochastic Game Elements

**Question:** Is YOMI Hustle fully deterministic given both players' inputs, or are there random elements (damage variance, etc.)?

If deterministic: simpler to reason about, but exploration in self-play can get stuck in cycles.
If stochastic: natural exploration from game variance, but harder to credit-assign.

---

## Experiment Plan

### Experiment 0: Baseline Measurement
- Run tournament: Gemini 3.1 Pro vs all scripted baselines (100 games each)
- Run tournament: Gemini 3.1 Pro vs other LLMs (Claude, GPT-4o, etc.)
- Establish Elo ratings and win rates
- Identify Gemini 3.1 Pro's weaknesses (if any)

### Experiment 1: SFT Warmup
- Collect 500+ winning games from Gemini 3.1 Pro
- SFT Qwen 2.5 7B on these trajectories via Tinker
- Evaluate SFT model vs scripted baselines
- **Success criterion:** SFT model beats `baseline/random` >90% and `baseline/scripted_safe` >60%

### Experiment 2: GRPO vs Scripted Baselines
- GRPO training with win/loss reward only
- Opponent pool: all 4 scripted baselines
- 100 training iterations, 50 rollouts per iteration
- **Success criterion:** >80% win rate vs all scripted baselines

### Experiment 3: Dense Reward Shaping
- Add HP delta to reward function
- Compare convergence speed vs Experiment 2
- **Success criterion:** Faster convergence, no reward hacking

### Experiment 4: Self-Play
- GRPO with self-play (train against frozen self)
- League of past checkpoints
- **Success criterion:** Elo increases over time, beats Experiment 2 policy

### Experiment 5: Targeted Anti-Gemini Training
- Collect Gemini 3.1 Pro trajectories
- Train against frozen Gemini policy
- **Success criterion:** >60% win rate vs Gemini 3.1 Pro over 100 games

### Experiment 6: Scale Up (If Needed)
- If 7B plateaus: try 14B, 32B
- If reward shaping is insufficient: add positional rewards
- If self-play diverges: add population-based training

---

## Architecture for RL Training

```
+------------------+     +------------------+     +------------------+
|  Tinker Cloud    |     |  Local Machine   |     |  Godot Instances |
|                  |     |                  |     |                  |
|  Training Client |<--->|  Training Loop   |<--->|  Game 1 (port    |
|  (GPU workers)   |     |  (Python)        |     |   8765)          |
|                  |     |                  |     |  Game 2 (port    |
|  Sampling Client |<--->|  Rollout         |     |   8766)          |
|  (inference)     |     |  Collector       |     |  ...             |
|                  |     |                  |     |  Game N (port    |
+------------------+     +------------------+     |   876N)          |
                                                  +------------------+
```

**Flow:**
1. Training loop requests rollouts
2. Sampling client generates actions for each game turn (via Tinker inference)
3. Actions sent to local Godot instances via daemon protocol
4. Game states returned, rewards computed
5. Trajectories assembled and sent to Tinker for training

**Alternative:** Use the existing daemon as the rollout collector. Add a `TinkerAdapter` alongside the existing `AnthropicAdapter`/`OpenAIAdapter` that calls Tinker's SamplingClient instead of a commercial API.

---

## OpenReward Considerations

### Could It Help?

OpenReward would help if we need to **scale game instances beyond what runs locally**. It provides:
- Managed environment hosting (autoscaled)
- Standard ORS protocol
- Direct Tinker integration for training

### The Packaging Problem

YOMI Hustle is a Godot game, not a Python environment. To use OpenReward, we'd need to:
1. Package the Godot runtime + game binary + mod into a Docker container
2. Run it headless inside OpenReward's sandbox
3. The Python ORS server wraps the WebSocket communication

This is **doable but significant engineering effort** that doesn't help with the core RL research question.

### Recommendation

**Skip OpenReward for now.** Start with local game instances + Tinker for training. The existing daemon architecture is already the right shape. Revisit OpenReward only if local rollout throughput becomes the bottleneck and you need distributed game instances.

The Steam dependency is not a platform issue (OpenReward has no Steam requirements) but a **packaging issue** - getting the game binary into a Docker container. This is orthogonal to the RL training itself.

---

## Key Decisions to Make

1. **Starting model:** Qwen 2.5 7B vs Llama 3.1 8B (check Tinker's model availability)
2. **Starting character:** Which YOMI Hustle character to train first?
3. **Reward function:** Win/loss only vs win/loss + HP delta from the start?
4. **Training opponents:** Scripted baselines only vs include LLM opponents early?
5. **Headless Godot:** Is `Engine.time_scale` acceleration feasible?
6. **SFT warmup:** Collect expert games first or go straight to RL?
7. **Observation format:** Full state vs curated compact state for RL?
