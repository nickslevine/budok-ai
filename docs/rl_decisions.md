---
name: rl-experiment-decisions
description: Key decisions made for RL training experiments - character selection, reward shape, training phases
type: project
---

## RL Experiment Decisions

**Character: Cowboy mirror matches** for all initial experiments.
**Why:** User's call. Cowboy has the largest moveset (44 moves, 38 with hitbox data), ranged mechanics (bullets), and is a good test of whether the model can learn resource management.
**How to apply:** All Phase 0-2 configs use `character_selection.mode: "assigned"` with both players as Cowboy.

**Reward shape: sparse (win/loss only) through Phase 2.**
**Why:** Cleanest signal, no reward hacking risk. Rejection sampling doesn't need shaping -- it's binary (winning trajectory or not). Add HP delta at low weight only if Phase 3 GRPO convergence is too slow.

**Training phases:**
1. Phase 0: Collect expert data (Gemini 3.1 Pro Cowboy mirrors, 200+ games)
2. Phase 1: SFT on winning Gemini trajectories (no game time)
3. Phase 2: Rejection sampling / expert iteration (moderate game time)
4. Phase 3: GRPO with match-level reward (only if Phase 2 plateaus)

**Model: Qwen 2.5 7B** as starting candidate (pending Tinker model availability check).
