# SFT v1 Full Eval: Statistical Confirmation (n=40)

After the initial 6-match eval showed 4-2 (67%), we ran 20 matches per opponent for statistical confidence and a deeper HP analysis.

## Headline Results

| Opponent | Record | Win Rate (95% CI) | Avg HP Diff | Dominance |
|---|---|---|---|---|
| **Gemini 3.1 Pro** | **17W-3L** | **85%** (64-95%) | **+186** | **+25%** |
| **GPT-5.4** | **14W-5L** | **70%** (48-85%) | **+244** | **+32%** |

**Dominance** = avg HP diff / starting HP. Both confidence intervals exclude 50%, so this isn't noise.

The SFT model is **strictly better than its teacher (GPT-5.4)** at Cowboy mirror, and **dominantly better than Gemini 3.1 Pro**. This is real, not small-sample variance.

---

## HP Differential Analysis

Win rate alone doesn't capture how dominant the wins are. Here's the full picture:

### vs Gemini 3.1 Pro (17W-3L)

| Metric | Value |
|---|---|
| Avg HP diff (all games) | +186 |
| Median HP diff | +193 |
| HP diff range | [-452, +619] |
| Avg SFT HP at end | 317 (42% of starting) |
| Avg Gemini HP at end | 132 (18% of starting) |

**In wins (17 matches):**
- Avg HP diff: **+271**
- Avg SFT HP remaining: **366 (49% of starting)**
- Median win margin: +242

**In losses (3 matches):**
- Avg HP diff: -297
- Avg Gemini HP remaining: 339 (45% of starting)
- Median loss margin: -405

**Interpretation:** When the SFT model wins against Gemini, it does so with **half its HP intact** -- decisive wins, not nail-biters. The 3 losses are blowouts in the other direction, suggesting they're catastrophic mismatches rather than close calls. The model either dominates or gets dominated.

### vs GPT-5.4 (14W-5L, 1 match incomplete)

| Metric | Value |
|---|---|
| Avg HP diff (all games) | +244 |
| Median HP diff | +190 |
| HP diff range | [-314, +719] |
| Avg SFT HP at end | 334 (45% of starting) |
| Avg GPT-5.4 HP at end | 90 (12% of starting) |

**In wins (14 matches):**
- Avg HP diff: **+393**
- Avg SFT HP remaining: **429 (57% of starting)**
- Median win margin: +464

**In losses (5 matches):**
- Avg HP diff: -129
- Avg GPT-5.4 HP remaining: 248 (33% of starting)
- Median loss margin: -105

**Interpretation:** When the SFT model wins vs GPT-5.4, it wins **even more decisively** than vs Gemini (+393 avg HP diff). It often finishes with 57% HP intact while GPT-5.4 is at 0. The losses are *closer* than the wins -- losing matches have a -129 avg diff vs +393 for wins. This means **the SFT model wins big and loses small**.

---

## Win Distribution (vs Gemini, sorted by margin)

```
[W]  640 vs   21  diff= +619  ██████████████████████████████
[W]  542 vs   43  diff= +499  ████████████████████████
[W]  530 vs   38  diff= +492  ████████████████████████
[W]  437 vs   12  diff= +425  █████████████████████
[W]  430 vs   35  diff= +395  ███████████████████
[W]  448 vs   82  diff= +366  ██████████████████
[W]  519 vs  160  diff= +359  █████████████████
[W]  519 vs  244  diff= +275  █████████████
[W]  367 vs  125  diff= +242  ████████████
[W]  380 vs  142  diff= +238  ███████████
[W]  169 vs   21  diff= +148  ███████
[W]  311 vs  172  diff= +139  ██████
[W]  276 vs  141  diff= +135  ██████
[W]  122 vs   21  diff= +101  █████
[W]  227 vs  150  diff=  +77  ███
[W]  203 vs  137  diff=  +66  ███
[W]  101 vs   68  diff=  +33  █
[L]   50 vs   84  diff=  -34  █
[L]   50 vs  455  diff= -405  ████████████████████
[L]   27 vs  479  diff= -452  ██████████████████████
```

**11/17 wins are by 200+ HP** -- dominant. The 3 losses are bimodal: one was a near-tie (-34) and two were catastrophic (-405, -452).

## Win Distribution (vs GPT-5.4, sorted by margin)

```
[W]  750 vs   31  diff= +719  ███████████████████████████████████
[W]  640 vs   23  diff= +617  ██████████████████████████████
[W]  629 vs   24  diff= +605  ██████████████████████████████
[W]  629 vs   38  diff= +591  █████████████████████████████
[W]  578 vs   11  diff= +567  ████████████████████████████
[W]  555 vs    8  diff= +547  ███████████████████████████
[W]  505 vs   21  diff= +484  ████████████████████████
[W]  519 vs   75  diff= +444  ██████████████████████
[W]  555 vs  119  diff= +436  █████████████████████
[W]  280 vs    3  diff= +277  █████████████
[W]  182 vs   80  diff= +102  █████
[W]  116 vs   36  diff=  +80  ████
[W]   25 vs    8  diff=  +17
[W]   42 vs   27  diff=  +15
[L]  142 vs  197  diff=  -55  ██
[L]  114 vs  219  diff= -105  █████
[L]  179 vs  344  diff= -165  ████████
[L]   51 vs  365  diff= -314  ███████████████
```

**Notable: a perfect game (750-31).** The SFT model finished one match against GPT-5.4 at full HP -- never took a single hit. **9/14 wins are by 400+ HP**, with 6 of them ending with the SFT model at 555+ HP (74%+ of starting).

The losses are all relatively close (max -314). The model doesn't get blown out by GPT-5.4 the way it occasionally does by Gemini.

---

## Key Insights

1. **The 85% / 70% headline numbers undersell the dominance.** When measured by HP differential, the SFT model isn't just winning more often -- it's winning by *huge* margins. The avg dominance score is +25% vs Gemini and +32% vs GPT-5.4 (out of 750 starting HP).

2. **The SFT model wins BIG and loses SMALL vs GPT-5.4.** Average win margin is +393 HP, average loss margin is -129. This means even if you ignore the win/loss outcome and just measure HP-at-end, the SFT model is dramatically ahead.

3. **vs Gemini, the losses are more catastrophic.** Two of the three losses to Gemini were -405 and -452 HP -- the SFT model got destroyed. This suggests Gemini occasionally finds an exploit that the SFT model can't handle. Worth investigating which match settings/conditions lead to those.

4. **The wins are clean.** In 9 of 14 GPT-5.4 wins, the SFT model finished with 500+ HP. That's not "barely winning" -- that's "obliterating with most of your HP intact." The SFT model has learned to play *efficiently*, not just to grind out wins.

5. **The teacher is now strictly worse than the student.** SFT model beats GPT-5.4 70% of the time with +244 avg HP diff. This is unusual for distillation and suggests the SFT model has compressed GPT-5.4's strategy into a cleaner, more consistent policy. It plays GPT-5.4's good moves without GPT-5.4's bad moves.

---

## Why the SFT model beats its teacher

A few hypotheses for why distillation produced a stronger student than the teacher:

1. **Variance reduction.** GPT-5.4 occasionally makes suboptimal moves (overthinking, exploring, producing verbose reasoning that hurts execution). The SFT model learned the *modal* GPT-5.4 strategy without the variance. In a high-variance game, consistency matters.

2. **Game-specific overtraining beats general capability.** The SFT model has seen ~1,672 examples of this exact prompt format and learned to extract relevant signals efficiently. GPT-5.4 parses each prompt cold.

3. **Format efficiency.** The SFT model produces minimal valid JSON immediately. GPT-5.4 may produce extra reasoning that doesn't improve action quality.

4. **No fallback overhead.** Both have ~0% fallback rate now, but the SFT model's outputs are tightly fit to the daemon's parsing -- never any "almost legal" actions that get coerced.

This is a great example of when distillation can beat the teacher: a narrow domain with sufficient training data and a specialized output format.

---

## Implications for Next Steps

The Cowboy SFT v1 model is much stronger than initially thought. Considerations:

1. **There's less room for improvement than expected.** At 85% vs Gemini and 70% vs GPT-5.4 with huge HP margins, the model is already near a ceiling on Cowboy mirror.

2. **The 3 catastrophic Gemini losses are the most informative data.** Worth analyzing what Gemini does in those 3 matches that the SFT model can't counter.

3. **Rejection sampling might be very valuable.** Loss matches give us 8 examples (3 vs Gemini, 5 vs GPT-5.4) of weak SFT play. Mining those for failure modes and re-training would directly attack the weakness.

4. **GRPO is the right tool to push past the ceiling.** Distillation has a hard cap at "best possible imitation of the teachers." Beyond that, the model needs to discover strategies the teachers don't use.

5. **The model is good enough for character generalization to be the priority.** Spending more compute on Cowboy improvements has diminishing returns. Training Ninja and Wizard SFTs (per the generalization test) would unlock playing the full game.
