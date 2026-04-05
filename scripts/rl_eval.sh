#!/usr/bin/env bash
# Evaluate an RL model against Gemini 3.1 Pro and/or GPT-5.4.
#
# Usage:
#   scripts/rl_eval.sh --model MODEL_ID [OPTIONS]
#
# Required:
#   --model MODEL_ID   OpenRouter model ID for the model under test
#                      (e.g. "qwen/qwen-2.5-7b-instruct" or a Tinker endpoint)
#
# Options:
#   --opponent OPPONENT   Opponent: "gemini", "gpt54", or "both" (default: both)
#   --num-matches N       Matches per opponent (default: 10)
#   --tag TAG             Tag for output directory (default: "eval")
#   --skip-mod-push       Skip mod packaging/push
#
# Output:
#   runs/rl_eval_<tag>/<timestamp>_<match_id>/  per-match artifacts
#   Prints HP differential summary at the end.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ─── Defaults ────────────────────────────────────────────────────────────────

MODEL_ID=""
OPPONENT="both"
NUM_MATCHES=10
TAG="eval"
EXTRA_ARGS=("--no-replay")

# ─── Parse arguments ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL_ID="$2"; shift 2 ;;
        --model=*)       MODEL_ID="${1#*=}"; shift ;;
        --opponent)      OPPONENT="$2"; shift 2 ;;
        --opponent=*)    OPPONENT="${1#*=}"; shift ;;
        --num-matches)   NUM_MATCHES="$2"; shift 2 ;;
        --num-matches=*) NUM_MATCHES="${1#*=}"; shift ;;
        --tag)           TAG="$2"; shift 2 ;;
        --tag=*)         TAG="${1#*=}"; shift ;;
        --skip-mod-push) EXTRA_ARGS+=("--skip-mod-push"); shift ;;
        --no-replay)     shift ;;  # already default
        -h|--help)       head -19 "$0" | tail -17; exit 0 ;;
        *)               EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$MODEL_ID" ]; then
    echo "ERROR: --model is required" >&2
    echo "Usage: scripts/rl_eval.sh --model MODEL_ID [--opponent gemini|gpt54|both] [--num-matches N]" >&2
    exit 1
fi

RUNS_DIR="$REPO_ROOT/runs/rl_eval_$TAG"
CONFIG_FILE="$REPO_ROOT/daemon/config/rl_eval_vs_gemini.json"
TEMP_CONFIG=$(mktemp /tmp/rl_eval_XXXXXX.json)

mkdir -p "$RUNS_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║  RL Evaluation                           ║"
echo "║  Model: $MODEL_ID"
echo "║  Opponent: $OPPONENT"
echo "║  Matches per opponent: $NUM_MATCHES"
echo "║  Output: $RUNS_DIR"
echo "╚══════════════════════════════════════════╝"
echo ""

# ─── Build opponent list ─────────────────────────────────────────────────────

OPPONENTS=()
case "$OPPONENT" in
    gemini) OPPONENTS=("google/gemini-3.1-pro-preview") ;;
    gpt54)  OPPONENTS=("openai/gpt-5.4") ;;
    both)   OPPONENTS=("google/gemini-3.1-pro-preview" "openai/gpt-5.4") ;;
    *)      echo "ERROR: --opponent must be gemini, gpt54, or both" >&2; exit 1 ;;
esac

# ─── Run matches ─────────────────────────────────────────────────────────────

cleanup_config() {
    rm -f "$TEMP_CONFIG"
}
trap cleanup_config EXIT

TOTAL_COMPLETED=0
TOTAL_FAILED=0

for OPP_ID in "${OPPONENTS[@]}"; do
    OPP_SHORT="${OPP_ID##*/}"
    echo ""
    echo "══════ vs $OPP_SHORT ══════"

    # Patch config: set model under test and opponent
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
# Set model under test
cfg['policies']['rl/model-under-test']['model'] = sys.argv[2]
# Set opponent
cfg['policy_mapping']['p2'] = sys.argv[3]
with open(sys.argv[4], 'w') as f:
    json.dump(cfg, f, indent=2)
" "$CONFIG_FILE" "$MODEL_ID" "$OPP_ID" "$TEMP_CONFIG"

    for i in $(seq 1 "$NUM_MATCHES"); do
        echo ""
        echo "  Match $i/$NUM_MATCHES vs $OPP_SHORT"

        if scripts/run_match.sh \
            --daemon-config "$TEMP_CONFIG" \
            --runs-root "$RUNS_DIR" \
            --trace-seed "$((i + 1000))" \
            "${EXTRA_ARGS[@]}"; then
            TOTAL_COMPLETED=$((TOTAL_COMPLETED + 1))
        else
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
            echo "  [eval] Match $i vs $OPP_SHORT failed, continuing..."
        fi
    done
done

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Evaluation Complete                     ║"
echo "║  Completed: $TOTAL_COMPLETED                          ║"
echo "║  Failed:    $TOTAL_FAILED                          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Collect results, print summary, and serialize to JSON
python3 -c "
import json, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

runs_dir = Path(sys.argv[1])
model_id = sys.argv[2]
results = []
for run_dir in sorted(runs_dir.iterdir()):
    result_file = run_dir / 'result.json'
    decisions_file = run_dir / 'decisions.jsonl'
    if not result_file.exists():
        continue
    result = json.loads(result_file.read_text())

    # Get final HP from last decision
    p1_hp, p2_hp = 0, 0
    if decisions_file.exists():
        lines = decisions_file.read_text().strip().split('\n')
        for line in reversed(lines):
            d = json.loads(line)
            obs = d.get('request_payload', {}).get('observation', {})
            fighters = obs.get('fighters', [])
            if len(fighters) == 2:
                p1_hp = fighters[0].get('hp', 0)
                p2_hp = fighters[1].get('hp', 0)
                break

    # p1 is our model, p2 is opponent
    winner = result.get('winner')
    hp_diff = p1_hp - p2_hp  # positive = we're ahead
    manifest_file = run_dir / 'manifest.json'
    opp_id = '?'
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        opp_id = manifest.get('policy_mapping', {}).get('p2', '?')

    results.append({
        'run_dir': str(run_dir.name),
        'opponent': opp_id,
        'winner': winner,
        'p1_hp': p1_hp,
        'p2_hp': p2_hp,
        'hp_diff': hp_diff,
        'won': winner == 'p1',
        'turns': result.get('total_turns', 0),
        'end_reason': result.get('end_reason', '?'),
    })

if not results:
    print('No results found.')
    sys.exit(0)

# Group by opponent
by_opp = defaultdict(list)
for r in results:
    by_opp[r['opponent']].append(r)

# Build summary for serialization
summary = {
    'model': model_id,
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'total_games': len(results),
    'opponents': {},
    'games': results,
}

for opp, games in sorted(by_opp.items()):
    wins = sum(1 for g in games if g['won'])
    losses = sum(1 for g in games if g['winner'] == 'p2')
    draws = len(games) - wins - losses
    avg_hp_diff = sum(g['hp_diff'] for g in games) / len(games)
    avg_turns = sum(g['turns'] for g in games) / len(games)

    summary['opponents'][opp] = {
        'games': len(games),
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': wins / len(games),
        'avg_hp_diff': round(avg_hp_diff, 1),
        'avg_turns': round(avg_turns, 1),
    }

# Write JSON results
output_path = runs_dir / 'eval_results.json'
with open(output_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Results saved to {output_path}')
print()

# Print summary to stdout
print('HP Differential Summary')
print('=' * 60)
for opp, games in sorted(by_opp.items()):
    s = summary['opponents'][opp]
    print()
    print(f'vs {opp}:')
    print(f'  Record: {s[\"wins\"]}W-{s[\"losses\"]}L-{s[\"draws\"]}D ({s[\"games\"]} games)')
    print(f'  Win rate: {s[\"win_rate\"]*100:.0f}%')
    print(f'  Avg HP diff: {s[\"avg_hp_diff\"]:+.0f} (positive = our model ahead)')
    print(f'  Avg turns: {s[\"avg_turns\"]:.0f}')
    print(f'  Per-game:')
    for g in games:
        marker = 'W' if g['won'] else ('L' if g['winner'] == 'p2' else 'D')
        print(f'    [{marker}] HP: {g[\"p1_hp\"]:>4d} vs {g[\"p2_hp\"]:>4d} (diff {g[\"hp_diff\"]:+d}, {g[\"turns\"]} turns)')
" "$RUNS_DIR" "$MODEL_ID"
