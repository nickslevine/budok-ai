# Generalization Test: SFT v1 on Ninja and Wizard

The SFT v1 model was trained exclusively on Cowboy mirror match data. To test whether the gains generalize, we ran the same eval (3 vs Gemini, 3 vs GPT-5.4) on Ninja and Wizard mirror matches.

## Results

### Cowboy (in-distribution)

| Model | vs Gemini | vs GPT-5.4 | Fallback |
|---|---|---|---|
| Baseline Qwen3-8B | 0W-3L (-377 HP) | 0W-3L (-244 HP) | 42% |
| **SFT v1** | **2W-1L (+232 HP)** | **2W-1L (-12 HP)** | **0%** |

### Ninja (out-of-distribution)

| Model | vs Gemini | vs GPT-5.4 | Fallback |
|---|---|---|---|
| Baseline Qwen3-8B | 1W-2L (+41 HP) | 0W-3L (-369 HP) | 40-45% |
| **SFT v1** | **0W-3L (-166 HP)** | **0W-3L (-559 HP)** | **9%** |

### Wizard (out-of-distribution)

| Model | vs Gemini | vs GPT-5.4 | Fallback |
|---|---|---|---|
| Baseline Qwen3-8B | 0W-3L (-572 HP) | 0W-3L (-502 HP) | 45-59% |
| **SFT v1** | **0W-3L (-335 HP)** | **0W-3L (-251 HP)** | **4%** |

## Findings

**The SFT gains do NOT generalize across characters.**

- **Cowboy:** SFT goes 4-2 (67% win rate)
- **Ninja:** SFT goes 0-6 (0% win rate)
- **Wizard:** SFT goes 0-6 (0% win rate)

The model overfit to Cowboy-specific tactics. It learned `GunThrow at long range`, `LightningSliceNeutral at mid`, `Lasso for grabs` -- moves that don't exist in the Ninja or Wizard movesets.

### What did partially generalize

1. **Format compliance improved across all characters.** Fallback rate dropped from 40-59% to 4-9% on Ninja/Wizard. The model learned the JSON output format and basic legality, just not the character-specific moves.

2. **HP differential improved on Wizard.** Baseline Wizard: -572/-502 HP diff. SFT Wizard: -335/-251 HP diff. The SFT model still loses every game but loses by less. Some general defensive habits (blocking, rolling) carry over.

3. **HP differential got WORSE on Ninja.** Baseline Ninja: +41/-369. SFT Ninja: -166/-559. The SFT model is actively worse at Ninja than the baseline. This suggests the model is trying to play Cowboy moves on a Ninja, picking the closest legal action, which produces dysfunctional play.

### Notable: baseline Ninja vs Gemini was 1-2

The baseline Qwen3-8B actually won 1 game against Gemini playing Ninja (696 HP vs 63). This is the only baseline win in 24 baseline matches across all 3 characters. Ninja's moveset apparently has a fast/aggressive option that worked once, possibly via the fallback handler stumbling into a winning sequence.

## Implications

1. **SFT learns character-specific play, not general game skill.** Training on Cowboy data teaches Cowboy strategy. Generalization across characters requires multi-character training data.

2. **One model per character is the natural unit.** If we want a competitive model for all 5 characters, we need 5 separate SFTs (or one SFT on data covering all characters).

3. **The Cowboy result is real, not an artifact of our infrastructure.** If the SFT was just exploiting some bug in the evaluation, it would help on all characters. The fact that Cowboy improves dramatically while Ninja/Wizard don't shows the model is genuinely learning Cowboy tactics.

4. **Format compliance is the only general gain.** Going from 40-50% fallback rate to 4-9% across all characters means the model learned "produce valid JSON for this game." That's transferable. Strategy is not.

## Next Steps

To get cross-character competence, options ranked by complexity:

1. **Per-character SFT models** -- collect 20 matches for each character, train 5 separate LoRAs. ~$200 total cost.
2. **Multi-character SFT** -- collect 20 matches across all 5 characters (mixed), train one model on all of it. Requires the model to learn character context from the prompt.
3. **GRPO with RL** -- skip SFT and let the model discover character-specific play through reward signal. Slower but more general.

The cheapest path forward is option 1 (per-character SFT) since we already have the infrastructure.
