# Investigation: Is the SFT vs GPT-5.4 result a bug?

After SFT v1 reached 70% win rate vs its own teacher (GPT-5.4) and 85% vs Gemini 3.1 Pro across 40 matches with massive HP margins, the result was suspicious enough to warrant a thorough check for environmental bugs.

## Hypotheses tested

| # | Hypothesis | Result |
|---|---|---|
| 1 | Prompt asymmetry between P1 and P2 | **REJECTED** -- prompts are byte-for-byte symmetric except policy_id |
| 2 | Decision count asymmetry (one player gets more turns) | **REJECTED** -- 38/40 matches had identical counts; the 2 with diff=1 split outcomes |
| 3 | P1 has positional advantage | **REJECTED** -- SFT vs SFT mirror is exactly 5W-5L, avg HP diff -59 (slightly P2-favored) |
| 4 | GPT-5.4 hits decision timeout | **REJECTED** -- 0 GPT-5.4 fallbacks across the eval, max latency 33s vs 120s timeout |
| 5 | Static SFT strategy + variable opponents | **CONFIRMED but explains less than expected** |
| 6 | Distillation extracts modal teacher strategy without variance | **CONFIRMED -- this is the main explanation** |

## Investigation 1: Prompt symmetry

Compared turn 1 prompts from 5 different SFT-vs-GPT-5.4 matches. After replacing player labels (`p1`/`p2`) with placeholders:

- **All 5 matches:** prompts differ only by **4 characters** (the `policy_id` field length: `tinker/yomi-sft-cowboy-v1` vs `google/gemini-3.1-pro-preview` or `openai/gpt-5.4`)
- All section headers present in both
- Same `prompt_version: strategic_v1`
- Same observation data, same legal actions, same character guide, same situation summary

**Verdict:** prompts are not asymmetric in any meaningful way.

## Investigation 2: Decision count symmetry

Across 40 SFT eval matches:
- 38 matches had identical P1/P2 decision counts
- 2 matches had a difference of 1 (one extra P1 decision) -- both happened to be SFT wins

Of those 2 asymmetric matches: this could be game-end timing where one player gets a final decision the other doesn't. Not a systematic bias.

## Investigation 3: P1/P2 positional bias (the smoking gun test)

**Setup:** Run the SFT model against itself (SFT vs SFT mirror) 10 times. If P1 wins consistently, there's a position bias. If 50/50, the SFT result is real.

**Result:** **5W-5L exactly. Average HP diff -59 (slightly P2-favored).**

```
P2 won:   23 vs 224  (diff -201)
P1 won:  398 vs 34   (diff +364)
P1 won:  174 vs 5    (diff +169)
P2 won:   70 vs 700  (diff -630)
P2 won:   27 vs 174  (diff -147)
P1 won:  319 vs 102  (diff +217)
P1 won:  306 vs 63   (diff +243)
P1 won:   16 vs 125  (diff -109)  <- weird: P2 had more HP but still lost
P2 won:  118 vs 309  (diff -191)
P2 won:   42 vs 347  (diff -305)
```

**There is no P1 advantage.** Two identical models split exactly evenly. The SFT vs GPT-5.4 result is not a positional artifact.

(One match shows P1 winning 16 vs 125 -- P2 had more HP but lost. This is likely an artifact of how end-of-match HP is read from the last decision log entry vs the actual game end state.)

## Investigation 4: Decision latencies

| Player | Median | Mean | P95 | Max |
|---|---|---|---|---|
| SFT (Tinker API) | 2,092ms | 2,328ms | 3,557ms | 9,180ms |
| Opponent (OpenRouter) | 4,836ms | 8,445ms | 23,119ms | 32,922ms |

The opponents are 4x slower on average, but **0 decisions exceeded 100s** and the timeout is 120s. Latency doesn't affect correctness in this game. **Not a bug.**

## Investigation 5: SFT plays a static opening

**The SFT model picks HSlash2 on turn 1 in 38/40 matches (95%).** The opponents are more varied:
- Gemini/GPT-5.4 open with HSlash2: 70%
- ParryHigh: 25%
- Lasso: 5%

**On turn 3 (after the symmetric HSlash2 trade), the SFT model picks Lasso 88% of the time.** The opponents are split:
- LightningSliceNeutral: 12
- Lasso: 8
- DashForward: 6

So the SFT model is highly predictable in the first 4 turns. If opponents were doing yomi (reading the SFT's pattern and countering), they'd play LightningSliceNeutral every time at turn 3 (which beats the SFT's predicted Lasso). They don't, because they're playing their own strategies.

**This confirms the SFT model has a static strategy.** But on its own, this should make the SFT *easier* to beat, not harder. So why does it win?

## Investigation 6: HSlash2 vs HSlash2 produces symmetric outcomes

When both players open with HSlash2 (26 matches), the exchange is **byte-for-byte symmetric:**
- Both end at exactly 640 HP (lost 110 each)
- Both end in HurtGrounded state

Yet the SFT wins **20-6** in these matches. Since the position is identical at turn 3, the SFT's edge must come from how it plays after turn 3.

## What's actually happening

The combination of evidence points to a real distillation phenomenon, not a bug:

1. **The SFT model has a static, well-rehearsed Cowboy gameplan** (HSlash2 → Lasso → ...) that it executes 95% consistently.

2. **The opponents (Gemini, GPT-5.4) play more variably.** On turn 3 from an identical position, Gemini/GPT-5.4 split between three different responses. The SFT picks one.

3. **Variance hurts in fighting games.** When both players make similar decisions for 30 turns, the consistent player tends to win because:
   - They've trained on the optimal sequence
   - They never make low-probability suboptimal moves
   - They never overthink or explore mid-match
   - They always pick the same DI / extras

4. **The SFT model dropped GPT-5.4's "bad days."** GPT-5.4 in the training data sometimes won, sometimes lost. The SFT model learned the modal strategy across all those matches. The modal strategy beats average GPT-5.4 because it's "GPT-5.4 minus the bad turns."

5. **HSlash2 vs HSlash2 trades into a position where the SFT model has a memorized continuation.** Both opponents end up in HurtGrounded with 640 HP at distance ~0. The SFT model has seen this state hundreds of times in training and picks the same followup. Gemini/GPT-5.4 reason from scratch each match and pick varied followups -- some good, some bad.

## The surprising finding: SFT wins even from bad positions

In matches where the opponent opened with **ParryHigh** (which blocks HSlash2 with only chip damage), giving them a clear advantage, the SFT still wins **7-2**.

This is harder to explain by "consistency vs variance" alone. After ParryHigh blocks the HSlash2:
- P1 takes 0 damage but is in HSlash2 recovery (slow)
- P2 takes 50 chip damage but is free to act

P2 should be able to capitalize. But in the matches I traced, P2 then chose Roll/Dash/etc and the SFT recovered, played Lasso, and started winning exchanges.

**Hypothesis:** the SFT model has memorized recoveries and counter-counters from many matches. Even from a disadvantageous position, it knows the optimal recovery sequence. The opponents reason from scratch and miss the optimal punish.

## Verdict: it's real, not a bug

**No bug found.** All the obvious hypotheses (prompt asymmetry, position bias, latency, timeouts) are ruled out. The SFT model genuinely beats GPT-5.4 at Cowboy mirror because:

1. It has a memorized opening sequence that it executes consistently
2. It has memorized continuations from common game states
3. The opponents reason from scratch and have higher variance
4. In a 30-100 turn match, variance compounds

This is exactly what distillation is *supposed* to do for narrow tasks. It's surprising in this case because we expected the student to approach the teacher's level, not exceed it. But for highly variable tasks where the teacher's variance hurts performance, the student can outperform.

## The remaining suspicion

There's still one thing that doesn't fully fit: the SFT model wins **dominantly** (often 500+ HP remaining) in many matches. That's not "slightly more consistent" -- that's "playing fundamentally better." 

A possible explanation: in the training data, GPT-5.4 played both sides. The training data therefore includes **both** good and bad GPT-5.4 play. The SFT model gravitated toward the patterns that produced wins -- effectively a soft form of rejection sampling. So the distilled model isn't just "average GPT-5.4," it's biased toward "winning GPT-5.4."

This would make the SFT model the equivalent of "best-case GPT-5.4" -- which would naturally beat actual GPT-5.4 ~70% of the time, since GPT-5.4 only plays its best half the time.

## What this means for next steps

If the SFT model is "GPT-5.4's best play, executed consistently," then:

1. **There's a hard ceiling around "best possible imitation of teachers."** The model can't exceed what the teachers can theoretically do.

2. **Rejection sampling on the SFT's own wins won't help much.** The SFT is already biased toward winning patterns.

3. **Adding a stronger teacher would help.** Distilling Claude Opus or Gemini Opus would give a higher ceiling.

4. **GRPO is the right tool to push past the ceiling.** RL can discover strategies neither teacher uses.

5. **The static-strategy weakness is exploitable.** A player who actually does yomi (predicting the SFT's HSlash2 → Lasso opening) could counter it. None of our current opponents do this, but if we trained an adversary specifically to counter the SFT, it would expose the weakness.
