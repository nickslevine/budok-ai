#!/usr/bin/env bash
# Collect SFT training data: GPT-5.4 Cowboy mirror matches.
#
# Usage:
#   scripts/rl_collect_sft.sh [NUM_MATCHES] [OPTIONS]
#
# Arguments:
#   NUM_MATCHES    Number of matches to run (default: 20)
#
# Options:
#   --skip-mod-push   Skip mod packaging/push (reuse existing)
#   --no-replay       Disable replay recording (faster)
#
# Output:
#   runs/rl_sft_cowboy/<timestamp>_<match_id>/  per-match artifacts

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NUM_MATCHES="${1:-20}"
shift 2>/dev/null || true

RUNS_DIR="$REPO_ROOT/runs/rl_sft_cowboy"
DAEMON_CONFIG="$REPO_ROOT/daemon/config/rl_sft_cowboy_mirror.json"

EXTRA_ARGS=("--no-replay")
for arg in "$@"; do
    case "$arg" in
        --skip-mod-push|--no-replay) EXTRA_ARGS+=("$arg") ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

mkdir -p "$RUNS_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║  RL SFT Data Collection                  ║"
echo "║  Model: GPT-5.4 vs GPT-5.4              ║"
echo "║  Character: Cowboy mirror                ║"
echo "║  Matches: $NUM_MATCHES                          ║"
echo "║  Output: $RUNS_DIR"
echo "╚══════════════════════════════════════════╝"
echo ""

COMPLETED=0
FAILED=0

for i in $(seq 1 "$NUM_MATCHES"); do
    echo ""
    echo "━━━ Match $i/$NUM_MATCHES ━━━"

    # Increment trace seed per match for variety
    MATCH_EXTRA_ARGS=("${EXTRA_ARGS[@]}")

    # Run match with --runs-root to organize output
    if scripts/run_match.sh \
        --daemon-config "$DAEMON_CONFIG" \
        --runs-root "$RUNS_DIR" \
        --trace-seed "$i" \
        "${MATCH_EXTRA_ARGS[@]}"; then
        COMPLETED=$((COMPLETED + 1))
    else
        FAILED=$((FAILED + 1))
        echo "[rl_collect] Match $i failed, continuing..."
    fi

    echo "[rl_collect] Progress: $COMPLETED completed, $FAILED failed, $((NUM_MATCHES - i)) remaining"
done

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Collection Complete                     ║"
echo "║  Completed: $COMPLETED                          ║"
echo "║  Failed:    $FAILED                          ║"
echo "║  Output:    $RUNS_DIR"
echo "╚══════════════════════════════════════════╝"

# Print summary of results
echo ""
echo "Match results:"
find "$RUNS_DIR" -name "result.json" -newer "$RUNS_DIR" | sort | while read -r f; do
    python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
status = r.get('status', '?')
winner = r.get('winner', 'draw')
turns = r.get('total_turns', '?')
print(f'  {status}: winner={winner}, turns={turns}')
" "$f" 2>/dev/null || true
done
