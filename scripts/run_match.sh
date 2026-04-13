#!/usr/bin/env bash
# End-to-end match runner for budok-ai on macOS + OrbStack.
#
# Automates: kill stale game, ensure Xvfb, package mod with correct bridge IP,
# push mod to VM, start daemon, launch game, wait for completion, report results.
#
# Usage:
#   scripts/run_match.sh [CONFIG_FILE] [OPTIONS]
#
# Arguments:
#   CONFIG_FILE              Path to a match.conf file (default: match.conf)
#
# Options (override config file values):
#   --daemon-config PATH     Daemon runtime config JSON
#   --log-level LVL          Log verbosity
#   --no-replay              Disable replay video recording (still saves match.replay)
#   --skip-mod-push          Skip mod packaging and push (use existing mod in VM)
#   --dry-run                Print what would be done, don't execute
#   -h, --help               Show this help
#
# Examples:
#   scripts/run_match.sh                           # use ./match.conf
#   scripts/run_match.sh my_match.conf             # use custom config
#   scripts/run_match.sh --daemon-config daemon/config/llm_first_test.json
#   scripts/run_match.sh --skip-mod-push           # reuse mod already in VM

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ─── Defaults ────────────────────────────────────────────────────────────────

VM_NAME="ubuntu"
VM_GAME_DIR="/home/$USER/games/yomi"
VM_DISPLAY=":99"
VM_RESOLUTION="1280x720"
DAEMON_CONFIG="daemon/config/llm_v_llm.json"
DAEMON_PORT=""
LOG_LEVEL="INFO"
RECORD_REPLAY=true
BRIDGE_IP_MODE="auto"
BRIDGE_IP="192.168.139.3"
ENV_FILE=".env"
SKIP_MOD_PUSH=false
DRY_RUN=false
RUNS_ROOT=""

# ─── Parse arguments ─────────────────────────────────────────────────────────

CONF_FILE=""
EXTRA_DAEMON_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --daemon-config)   DAEMON_CONFIG="$2"; shift 2 ;;
        --daemon-config=*) DAEMON_CONFIG="${1#*=}"; shift ;;
        --log-level)       LOG_LEVEL="$2"; shift 2 ;;
        --log-level=*)     LOG_LEVEL="${1#*=}"; shift ;;
        --no-replay)       RECORD_REPLAY=false; shift ;;
        --skip-mod-push)   SKIP_MOD_PUSH=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --match-history)   EXTRA_DAEMON_ARGS+=("--match-history" "$2"); shift 2 ;;
        --match-history=*) EXTRA_DAEMON_ARGS+=("--match-history" "${1#*=}"); shift ;;
        --runs-root)       EXTRA_DAEMON_ARGS+=("--runs-root" "$2"); RUNS_ROOT="$2"; shift 2 ;;
        --runs-root=*)     EXTRA_DAEMON_ARGS+=("--runs-root" "${1#*=}"); RUNS_ROOT="${1#*=}"; shift ;;
        --trace-seed)      EXTRA_DAEMON_ARGS+=("--trace-seed" "$2"); shift 2 ;;
        --trace-seed=*)    EXTRA_DAEMON_ARGS+=("--trace-seed" "${1#*=}"); shift ;;
        -h|--help)         head -25 "$0" | tail -23; exit 0 ;;
        -*)                EXTRA_DAEMON_ARGS+=("$1"); shift ;;
        *)
            if [ -z "$CONF_FILE" ]; then
                CONF_FILE="$1"
            else
                EXTRA_DAEMON_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# ─── Load config file ────────────────────────────────────────────────────────

if [ -z "$CONF_FILE" ]; then
    # Default: look for match.conf in repo root
    if [ -f "$REPO_ROOT/match.conf" ]; then
        CONF_FILE="$REPO_ROOT/match.conf"
    fi
fi

if [ -n "$CONF_FILE" ]; then
    if [ ! -f "$CONF_FILE" ]; then
        printf 'ERROR: Config file not found: %s\n' "$CONF_FILE" >&2
        exit 1
    fi
    printf 'Loading config: %s\n' "$CONF_FILE"
    # Source only KEY=VALUE lines, skip comments and blanks
    while IFS='=' read -r key value; do
        # Skip comments and blank lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        # Trim whitespace
        key="$(echo "$key" | xargs)"
        value="$(echo "$value" | xargs)"
        [ -z "$key" ] && continue
        # Export the variable (CLI args already set take precedence via order)
        eval "$key=\"$value\""
    done < "$CONF_FILE"
fi

# ─── Resolve paths ───────────────────────────────────────────────────────────

# Make relative paths absolute from repo root
[[ "$DAEMON_CONFIG" != /* ]] && DAEMON_CONFIG="$REPO_ROOT/$DAEMON_CONFIG"
[[ "$ENV_FILE" != /* ]] && ENV_FILE="$REPO_ROOT/$ENV_FILE"

# ─── Utility functions ───────────────────────────────────────────────────────

log() { printf '[run_match] %s\n' "$*"; }
err() { printf '[run_match] ERROR: %s\n' "$*" >&2; }
warn() { printf '[run_match] WARNING: %s\n' "$*" >&2; }

run_vm() {
    # Run a command in the OrbStack VM
    orb run -m "$VM_NAME" bash -c "$1"
}

# ─── Pre-flight checks ───────────────────────────────────────────────────────

log "Pre-flight checks..."

# Check required tools
for cmd in uv orb; do
    if ! command -v "$cmd" &>/dev/null; then
        err "$cmd is not installed"
        exit 1
    fi
done

# Check daemon config exists
if [ ! -f "$DAEMON_CONFIG" ]; then
    err "Daemon config not found: $DAEMON_CONFIG"
    exit 1
fi

# Load environment variables (.env file)
if [ -f "$ENV_FILE" ]; then
    log "Loading env from $ENV_FILE"
    set -a
    # Only source non-comment, non-empty lines
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        eval "export $line" 2>/dev/null || true
    done < "$ENV_FILE"
    set +a
else
    warn "No .env file found at $ENV_FILE"
fi

# Check VM exists and start it if stopped
if ! orb list 2>/dev/null | grep -q "$VM_NAME"; then
    err "OrbStack VM '$VM_NAME' not found. Create it first (see docs/macos.md)"
    exit 1
fi
if orb list 2>/dev/null | grep "$VM_NAME" | grep -q "stopped"; then
    log "VM '$VM_NAME' is stopped, starting it..."
    orb start "$VM_NAME"
    sleep 2
fi

# Check game exists in VM
if ! run_vm "test -f $VM_GAME_DIR/YourOnlyMoveIsHUSTLE.x86_64" 2>/dev/null; then
    err "Game not found at $VM_GAME_DIR in VM '$VM_NAME'"
    exit 1
fi

# ─── Resolve bridge IP ───────────────────────────────────────────────────────

if [ "$BRIDGE_IP_MODE" = "auto" ]; then
    BRIDGE_IP=$(ifconfig bridge100 2>/dev/null | grep "inet " | awk '{print $2}' || true)
    if [ -z "$BRIDGE_IP" ]; then
        err "Could not detect bridge100 IP. Is OrbStack running?"
        err "Set BRIDGE_IP_MODE=manual and BRIDGE_IP=<ip> in your config."
        exit 1
    fi
    log "Detected bridge IP: $BRIDGE_IP"
else
    log "Using manual bridge IP: $BRIDGE_IP"
fi

if [ "$DRY_RUN" = true ]; then
    log "DRY RUN — would execute the following steps:"
    log "  1. Kill stale game processes in VM"
    log "  2. Ensure Xvfb on display $VM_DISPLAY"
    log "  3. Package mod with bridge IP $BRIDGE_IP, push to VM"
    log "  4. Start daemon with config $DAEMON_CONFIG"
    log "  5. Launch game in VM"
    log "  6. Wait for match completion"
    exit 0
fi

# ─── Step 1: Kill stale game processes ────────────────────────────────────────

log "Killing stale game processes in VM..."
run_vm "ps aux | grep YourOnly | grep -v grep | awk '{print \$2}' | xargs -r kill -9" 2>/dev/null || true

# ─── Step 2: Ensure Xvfb is running ──────────────────────────────────────────

log "Ensuring Xvfb on display $VM_DISPLAY..."
if ! run_vm "DISPLAY=$VM_DISPLAY xdpyinfo &>/dev/null" 2>/dev/null; then
    log "Starting Xvfb..."
    run_vm "Xvfb $VM_DISPLAY -screen 0 ${VM_RESOLUTION}x24 -nocursor &>/dev/null &"
    sleep 1
    if ! run_vm "DISPLAY=$VM_DISPLAY xdpyinfo &>/dev/null" 2>/dev/null; then
        # Xvfb may have failed due to stale lock files
        run_vm "rm -f /tmp/.X*-lock /tmp/.X11-unix/X*" 2>/dev/null || true
        run_vm "Xvfb $VM_DISPLAY -screen 0 ${VM_RESOLUTION}x24 -nocursor &>/dev/null &"
        sleep 1
    fi
fi

# ─── Step 3: Package mod with correct bridge IP and push to VM ────────────────

if [ "$SKIP_MOD_PUSH" = false ]; then
    log "Packaging mod with bridge IP $BRIDGE_IP..."

    # Temporarily set the mod config to point at the bridge IP
    MOD_CONFIG="$REPO_ROOT/mod/YomiLLMBridge/config/default_config.json"
    ORIGINAL_HOST=$(python3 -c "import json; print(json.load(open('$MOD_CONFIG'))['transport']['host'])")

    # Update mod config to use bridge IP
    python3 -c "
import json
with open('$MOD_CONFIG') as f:
    cfg = json.load(f)
cfg['transport']['host'] = '$BRIDGE_IP'
with open('$MOD_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)
    f.write('\n')
"

    # Package the mod
    scripts/package_mod.sh

    # Restore original mod config
    python3 -c "
import json
with open('$MOD_CONFIG') as f:
    cfg = json.load(f)
cfg['transport']['host'] = '$ORIGINAL_HOST'
with open('$MOD_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)
    f.write('\n')
"

    # Push to VM
    log "Pushing mod to VM..."
    run_vm "mkdir -p $VM_GAME_DIR/mods"
    orb push -m "$VM_NAME" "$REPO_ROOT/dist/YomiLLMBridge.zip" "$VM_GAME_DIR/mods/YomiLLMBridge.zip"

    log "Mod installed in VM"
else
    log "Skipping mod push (--skip-mod-push)"
fi

# ─── Step 4: Start daemon ────────────────────────────────────────────────────

log "Starting daemon..."

DAEMON_ARGS=("--config" "$DAEMON_CONFIG" "--host" "0.0.0.0" "--log-level" "$LOG_LEVEL")

if [ -n "$DAEMON_PORT" ]; then
    DAEMON_ARGS+=("--port" "$DAEMON_PORT")
fi

if [ "$RECORD_REPLAY" = "false" ]; then
    DAEMON_ARGS+=("--no-record-replay")
elif [ "$RECORD_REPLAY" = "true" ]; then
    DAEMON_ARGS+=("--record-replay")
fi

DAEMON_ARGS+=("--replay-vm" "$VM_NAME" "--replay-display" "$VM_DISPLAY")
DAEMON_ARGS+=("${EXTRA_DAEMON_ARGS[@]+"${EXTRA_DAEMON_ARGS[@]}"}")

DAEMON_PID=""
GAME_PID_FILE=$(mktemp)

cleanup() {
    if [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        log "Stopping daemon (pid $DAEMON_PID)..."
        kill "$DAEMON_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$DAEMON_PID" 2>/dev/null || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
    # Kill game process in VM
    run_vm "ps aux | grep YourOnly | grep -v grep | awk '{print \$2}' | xargs -r kill -9" 2>/dev/null || true
    rm -f "$GAME_PID_FILE"
}
trap cleanup EXIT

# Record start time so we only look for run dirs created after this point
MATCH_START_EPOCH=$(date +%s)

uv run --project daemon yomi-daemon "${DAEMON_ARGS[@]}" &
DAEMON_PID=$!

# Wait for daemon to start listening
LISTEN_PORT="${DAEMON_PORT:-8765}"
log "Waiting for daemon to listen on port $LISTEN_PORT..."
for i in $(seq 1 15); do
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
        err "Daemon exited unexpectedly"
        exit 1
    fi
    if python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
s.connect(('127.0.0.1', int(sys.argv[1])))
s.close()
" "$LISTEN_PORT" 2>/dev/null; then
        break
    fi
    sleep 1
done

if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    err "Daemon exited before accepting connections"
    exit 1
fi

log "Daemon listening on ws://0.0.0.0:$LISTEN_PORT"

# ─── Step 5: Launch game in VM ───────────────────────────────────────────────

log "Launching game in VM..."

run_vm "
export LIBGL_ALWAYS_SOFTWARE=1
export LD_LIBRARY_PATH=$VM_GAME_DIR:\$LD_LIBRARY_PATH
DISPLAY=$VM_DISPLAY $VM_GAME_DIR/YourOnlyMoveIsHUSTLE.x86_64 &>/dev/null &
echo \$!
" > "$GAME_PID_FILE" 2>/dev/null || true

log "Game launched. Waiting for match to complete (Ctrl+C to stop)..."

# ─── Step 6: Wait for match completion ────────────────────────────────────────

# The daemon has a known shutdown hang (websockets wait_closed deadlock).
# We poll: once a result.json appears in the latest run dir, the match is done.
# Give extra time for replay recording, then kill the daemon.

log "Polling for match result..."

RESULT_FILE=""
while true; do
    # Check if daemon died unexpectedly (before any result)
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
        break
    fi

    # Look for a result.json in a run dir created AFTER this script started
    POLL_DIR="${RUNS_ROOT:-runs}"
    LATEST_RUN=""
    for d in $(ls -td "$POLL_DIR"/*/ 2>/dev/null); do
        dir_epoch=$(stat -f %m "$d" 2>/dev/null || stat -c %Y "$d" 2>/dev/null || echo 0)
        if [ "$dir_epoch" -ge "$MATCH_START_EPOCH" ]; then
            LATEST_RUN="$d"
            break
        fi
    done
    if [ -n "$LATEST_RUN" ] && [ -f "${LATEST_RUN}result.json" ] && \
       python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('status') in ('completed','failed') else 1)" "${LATEST_RUN}result.json" 2>/dev/null; then
        RESULT_FILE="${LATEST_RUN}result.json"
        if [ "$RECORD_REPLAY" = "true" ]; then
            # Result found — wait for replay video (up to 3 min for recording + pull)
            log "Match result found, waiting for replay video recording..."
            for i in $(seq 1 180); do
                if [ -f "${LATEST_RUN}replay.mp4" ]; then
                    log "Replay video ready"
                    break
                fi
                if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            # Give daemon a few seconds to finish cleanup, then kill it
            sleep 3
        else
            # Still wait briefly for the saved .replay file to be pulled, but do
            # not wait for replay playback / ffmpeg recording.
            log "Match result found, waiting briefly for replay file..."
            for i in $(seq 1 15); do
                if [ -f "${LATEST_RUN}match.replay" ]; then
                    log "Replay file ready"
                    break
                fi
                if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            sleep 1
        fi
        break
    fi
    sleep 2
done

# Kill the daemon if it's still hanging
if kill -0 "$DAEMON_PID" 2>/dev/null; then
    log "Stopping daemon..."
    kill "$DAEMON_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$DAEMON_PID" 2>/dev/null || true
fi
wait "$DAEMON_PID" 2>/dev/null || true
DAEMON_PID=""

# Report results
if [ -n "$RESULT_FILE" ]; then
    MATCH_STATUS=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('status','?'))" 2>/dev/null || echo "?")
    if [ "$MATCH_STATUS" = "failed" ]; then
        log "Match failed (game disconnected or crashed)"
    else
        log "Match completed successfully!"
    fi
    log "Artifacts: $LATEST_RUN"
    printf '\n'
    printf '╔══════════════════════════════════╗\n'
    printf '║         MATCH RESULT             ║\n'
    printf '╠══════════════════════════════════╣\n'
    python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
print(f'║  Status:  {r.get(\"status\", \"unknown\"):<21s} ║')
print(f'║  Winner:  {r.get(\"winner\", \"unknown\"):<21s} ║')
print(f'║  Reason:  {r.get(\"end_reason\", \"unknown\"):<21s} ║')
print(f'║  Turns:   {str(r.get(\"total_turns\", \"unknown\")):<21s} ║')
" "$RESULT_FILE" 2>/dev/null || true
    printf '╚══════════════════════════════════╝\n'
    if [ -f "${LATEST_RUN}match.replay" ]; then
        log "Replay file: ${LATEST_RUN}match.replay"
    fi
    if [ -f "${LATEST_RUN}replay.mp4" ]; then
        log "Replay video: ${LATEST_RUN}replay.mp4"
    fi
else
    err "Match did not complete — no result.json found"
    exit 1
fi
