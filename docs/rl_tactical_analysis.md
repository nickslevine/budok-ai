# Tactical Analysis: Baseline vs SFT v1

## Overview

Analysis of Qwen3-8B's tactical behavior before and after SFT, based on 6 matches each (3 vs Gemini 3.1 Pro, 3 vs GPT-5.4). Cowboy mirror matches, 750 HP.

| Metric | Baseline | SFT v1 |
|---|---|---|
| Record | 0W-6L | 4W-2L |
| Total P1 decisions | 378 | 264 |
| Fallback rate | 42.3% | 0.4% |
| Unique moves used | 28 (real only) | 28 |

---

## 1. Action Frequency Shift

### Baseline (real decisions only, excluding fallbacks)

| Action | % | Role |
|---|---|---|
| Lasso | 19.3% | Long-range grab |
| Fall | 17.9% | Aerial descent/approach |
| HSlash2 | 8.7% | Mid-range sword normal |
| DashForward | 8.3% | Approach |
| HAirSlash | 5.0% | Aerial attack |
| Pommel | 5.0% | Close-range poke |

### SFT v1 (all decisions, since fallback rate is ~0%)

| Action | % | Role |
|---|---|---|
| Continue | 30.3% | Mid-combo continuation (often forced) |
| ParryHigh | 8.3% | Block/defense |
| GunThrow | 6.8% | Long-range projectile |
| LightningSliceNeutral | 6.8% | Mid-range lunging attack |
| Lasso | 6.4% | Long-range grab |
| Roll | 4.9% | Dodge |
| HSlash2 | 3.4% | Mid-range sword normal |
| Grab | 3.4% | Close-range throw |

### Key changes

**Defense emerged.** Baseline Qwen made defensive choices on <3% of its real decisions. All observed blocking/parrying (ParryHigh, ParrySuper, ParryAfterWhiff) was from the fallback handler, not the model. After SFT, ParryHigh is the model's #2 most-used move (8.3%), Roll appears at 4.9%, and blocking is a deliberate choice. The model learned that defense is part of the game.

**Ranged tools adopted.** Baseline never used GunThrow or LightningSlice as primary tools. SFT uses GunThrow as its primary long-range option (6.8%, almost exclusively at >250 units) and LightningSliceNeutral as its primary mid-range poke (6.8%, concentrated at 100-250 units). This is correct Cowboy play.

**Lasso addiction reduced.** Baseline Qwen used Lasso on 19.3% of real decisions -- its signature move at every range. SFT reduced this to 6.4%, properly contextualizing it as a mid-range option rather than a universal answer.

**Grab mixup learned.** Baseline never used Grab intentionally. SFT uses Grab at 3.4%, concentrated at close range (9.7% of close-range decisions). This creates a classic attack/grab/block mixup that the baseline lacked.

---

## 2. Range Awareness

### Baseline

| Range | Primary actions |
|---|---|
| Close (<100) | Pommel, HSlash2, Lasso |
| Mid (100-250) | Lasso, DashForward, HSlash2 |
| Long (>250) | Fall, Lasso, DashForward |

Baseline had reasonable range awareness but used Lasso at every range. At long range, 58% of choices were pure movement (approach tools).

### SFT v1

| Range | Primary actions |
|---|---|
| Close (<100) | ParryHigh (14%), Grab (9.7%), Roll (8.6%), UpwardSwipe (6.5%) |
| Mid (100-250) | LightningSliceNeutral (15.4%), Lasso (13.7%), ParryHigh (7.7%), HSlash2 (6%) |
| Long (>250) | GunThrow (29.6%), DoubleJump (13%), ImpaleTeleport (9.3%), Dash (9.3%) |

**Dramatically more range-differentiated.** Each range bracket has distinct tools:
- Close: defensive/mixup (ParryHigh + Grab + Roll)
- Mid: poking tools (LightningSlice + Lasso + HSlash2)
- Long: projectiles and approach (GunThrow + movement)

---

## 3. Offense/Defense Balance

| Category | Baseline | SFT v1 |
|---|---|---|
| Offensive | 57.8% | 42.0% |
| Defensive | 2.8% | 52.3%* |
| Movement | 36.2% | 5.7% |

*SFT's defensive % is inflated by Continue (30.3%, often forced mid-combo). Excluding Continue: ~60% offense, ~30% defense, ~10% movement.

The SFT model plays a more balanced style. Baseline was nearly pure offense (58%) with almost no defense. SFT maintains offense as the primary mode but integrates meaningful defensive play (~30% when initiating), creating a rock-paper-scissors dynamic that the baseline lacked.

---

## 4. Adaptation and Pattern Recognition

### Baseline
- After taking hits: shifts toward approach tools (Lasso, DashForward) -- likely retreating to reset
- When losing: slightly more aggressive (more DashForward, less Fall)
- When winning: curiously conservative (more Fall, less attacking)
- 13.7% consecutive action repetition rate (mostly Fall sequences)

### SFT v1
- After opponent GunThrow: responds with Dash (44%) or LightningSlice (22%) -- closing distance through the gap
- After opponent Lasso: responds with LightningSlice (50%) -- beating Lasso with a faster attack
- After opponent ParryHigh: responds with Lasso (50%) or AirGrab (17%) -- grabs beat blocks
- After opponent HSlash2: responds with Lasso (29%) or ParryHigh (29%) -- grab or block
- When winning: more ParryHigh and Grab (controlled play)
- When losing: more GunThrow, DoubleJump, Fall (desperate mobility)
- Near-zero consecutive repetition of non-Continue moves

**The SFT model learned matchup-specific counterplay.** It responds to blocks with grabs, to grabs with attacks, and to attacks with blocks -- the core RPS dynamic of fighting games. The baseline model didn't show this pattern.

---

## 5. DI (Directional Influence) Usage

### Baseline
Not analyzed in detail, but the baseline model's DI was likely default (0,0) since most decisions came through the fallback handler which sets DI to (0,0).

### SFT v1
- **87.7% of decisions use non-zero DI** (199/227 with DI data)
- Common patterns: (100,0) for strong horizontal escape, (20,0) for forward lean, (-100,0) for backward escape
- **34 unique DI values** used on Continue decisions alone
- DI varies by situation: aerial moves use outward/upward, TechRoll uses decisive ±100 horizontal
- Feint and Reverse flags are never used (0%) -- a gap

The SFT model learned meaningful DI from the GPT-5.4 training data. This is a significant hidden advantage -- even on forced Continue turns, the model is actively influencing its trajectory.

---

## 6. Reasoning Quality

### Baseline
- 216/218 real decisions have reasoning text (~169 chars avg)
- Quality is surprisingly coherent: references range, history, anti-repetition
- Occasional factual errors (wrong range estimates)
- Example: "At 258 units most melee options whiff; Gun Throw is one of the few in-range threats"

### SFT v1
- Nearly all decisions have reasoning text
- More detailed and tactically specific
- Explicitly references opponent tendencies and matchup logic
- Example: "Close neutral at 116 strongly invites an immediate attack or Lightning Slice; blocking is a strong mixup here and avoids another dash/forward commitment while opponent has initiative."
- 24.4% of reasonings reference history/patterns; 21.4% mention anti-repetition

Both models produce coherent reasoning. The SFT model's reasoning is slightly more sophisticated -- it references initiative, commitment levels, and opponent reads more frequently.

---

## 7. Coverage Gaps (Both Models)

Both models use ~28 unique moves out of ~53 available. Notable unused moves:

| Never used | Why it matters |
|---|---|
| 3Combo | Basic combo starter, high damage potential |
| SpotDodge | Invincible frames, strong defensive option |
| ForesightNeutral | Time marker for mixups |
| FastTeleport | Repositioning super |
| VSlash | Vertical sword attack |
| BackSlash | Cross-up attack |
| 1000Cuts | High-damage super |
| DefensiveBurst | Emergency escape |
| Brandish/Shoot2 | Quick Draw stance + gun combos |

The SFT model inherited GPT-5.4's move preferences. If GPT-5.4 never uses 3Combo or SpotDodge in training data, the SFT model won't learn them either. Expanding the training data to include more diverse play (or using RL to discover these moves) could unlock additional performance.

---

## Summary

The SFT didn't just fix format compliance -- it fundamentally changed how the model plays:

1. **Learned defense:** 3% -> 30% defensive play when initiating
2. **Learned range-appropriate tools:** GunThrow at long range, LightningSlice at mid, Grab at close
3. **Learned the RPS dynamic:** responds to blocks with grabs, grabs with attacks, attacks with blocks
4. **Learned meaningful DI:** 88% non-zero DI with situational variation
5. **Reduced Lasso fixation:** 19% -> 6%, properly contextualized
6. **Maintained diversity:** 28 unique moves, near-zero spamming
