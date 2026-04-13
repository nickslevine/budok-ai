# Data Flow

This document traces a single turn from game signal to applied action, covering every component boundary crossing. Read this first to understand how the system works end to end.

## Overview

```
Game (Godot)                    Daemon (Python)
─────────────                   ───────────────
player_actionable signal
  │
  ▼
TurnHook.gd
  ├─ IdentifierFactory.gd ──► match_id, turn_id
  ├─ ObservationBuilder.gd ──► observation dict
  ├─ LegalActionBuilder.gd ──► legal_actions list
  │
  ▼
ProtocolCodec.gd
  └─ wraps in v2 envelope
  │
  ▼
BridgeClient.gd ─── WebSocket ───► server.py
                                      │
                                      ▼
                                   parse_envelope()
                                      │
                                      ▼
                                   _handle_decision_request()
                                      │
                                      ├─ player_id → adapter lookup
                                      │
                                      ▼
                                   resolve_adapter_decision()
                                      │
                                      ├─ adapter.decide_with_trace()
                                      │    ├─ prompt.py renders template
                                      │    ├─ provider API call (or baseline logic)
                                      │    └─ response_parser.py validates
                                      │
                                      ├─ (on failure) fallback.py
                                      │
                                      ▼
                                   ActionDecision in v2 envelope
                                      │
BridgeClient.gd ◄── WebSocket ────────┘
  │
  ▼
DecisionValidator.gd
  │
  ▼
ActionApplier.gd
  └─ on_action_selected(action, data, extra)
  │
  ▼
Telemetry.gd ──► event envelope ──► server.py ──► storage/writer.py
```

## Phase 1: Game Signal (Mod)

The game emits `player_actionable` on `Global.current_game` when a player can act. `TurnHook.gd` is connected to this signal and coordinates the turn request.

`RuntimeCompatibility.gd` gates this entire flow. Before the bridge sends any turn requests, it checks:

- `Global.VERSION == "1.9.20-steam"`
- Engine version prefix `3.5.1`
- Expected game signals exist (`player_actionable`, `game_ended`, `game_won`)
- Both fighter objects expose required fields

If any check fails, the bridge does not activate.

## Phase 2: State Capture (Mod)

`TurnHook.gd` orchestrates three builders:

1. **`IdentifierFactory.gd`** mints `match_id` (once per match) and `turn_id` (incrementing per turn).

2. **`ObservationBuilder.gd`** serializes deterministic game state from live objects:
   - Both fighters: position, velocity, health, meter, burst, facing, current state, actionability flags, and optional character-specific data
   - Character name resolved via `fighter.filename` reverse-lookup against known scene paths (`SCENE_PATH_TO_CHARACTER`), since `game.gd` overwrites fighter node names to `"P1"`/`"P2"` at runtime
   - Active projectiles, installs, and effects (classified by `OBJECT_CATEGORY_MAP`)
   - Stage bounds
   - Turn history (last 10 entries, enriched with both players' actions, HP, and positions per turn)
   - Tick and frame counters

3. **`LegalActionBuilder.gd`** enumerates actions from the UI-backed action button containers (`P1ActionButtons` / `P2ActionButtons`):
   - Reads each visible button's `action_name`, `state`, and `data_ui_scene`
   - Expands payload parameters into structured `payload_spec` objects
   - Records `supports` flags (DI, feint, reverse, prediction) and tactical metadata (damage, startup frames, range, meter cost)

Both builders produce deterministic output. Keys are sorted before hashing. `ProtocolCodec.gd` computes `state_hash` and `legal_actions_hash` as SHA-256 digests of canonical JSON.

## Phase 3: Transport (Mod → Daemon)

`ProtocolCodec.gd` wraps the `DecisionRequest` in a v2 envelope:

```json
{
  "type": "decision_request",
  "version": "v2",
  "ts": "2026-03-12T00:00:00Z",
  "payload": { ... }
}
```

`BridgeClient.gd` sends this as a text WebSocket frame to the daemon.

## Phase 4: Handshake (First Connection Only)

Before any turn requests flow, the mod and daemon perform a one-time handshake:

1. Mod sends `hello` envelope with `game_version`, `mod_version`, `schema_version`, `supported_protocol_versions`, and optional `auth_token`.
2. `server.py` validates the envelope, checks auth (if configured), negotiates protocol version, and confirms schema version.
3. Daemon responds with `hello_ack` containing `accepted_protocol_version`, `daemon_version`, `policy_mapping`, and optional `config` snapshot.
4. A `MatchSession` is created to track session state.

## Phase 5: Routing (Daemon)

`server.py._run_match_loop()` receives envelopes from the WebSocket and routes by `type`:

- **`decision_request`** → `_handle_decision_request()` (dispatched concurrently via `asyncio.create_task`)
- **`event`** → logged to artifact writer
- **`match_ended`** → waits for in-flight decision tasks, then triggers artifact finalization and session cleanup

Decision requests for both players are processed concurrently. When two `decision_request` envelopes arrive back-to-back (one per player), their LLM API calls run in parallel, roughly halving per-turn-pair latency compared to sequential processing.

On the first `decision_request`, the server lazily initializes:
- `MatchManifest` with pinned config, versions, and seed
- `MatchArtifactWriter` that creates the `runs/<timestamp>_<match_id>/` directory

## Phase 6: Decision Resolution (Daemon)

`_handle_decision_request()` maps `player_id` to a `PlayerSlot` (`p1`/`p2`), looks up the assigned `PolicyAdapter`, and calls `resolve_adapter_decision()`.

### Provider-backed path

For provider adapters (Anthropic, OpenAI, OpenRouter):

1. `adapter.decide_with_trace(request)` is called.
2. Inside the adapter, `prompt.py:render_prompt()` assembles the full prompt from the template body, output contract, turn context, situation summary, observation, and legal actions. The situation summary pre-computes distance, range label, HP comparison, and repetition warnings. Legal actions are enriched with move catalog metadata (category, speed, damage, range) from `prompts/move_catalog.json`.
3. The adapter calls the provider API with the rendered prompt.
4. The raw response is parsed by `response_parser.py:parse_action_decision_with_correction()`:
   - Tries structured parsing (mapping/ProtocolModel)
   - Falls back to JSON extraction from text
   - Falls back to key-value text parsing
   - Normalizes defaults (`match_id`, `turn_id`, `data`, `extra`)
   - Strips `extra.prediction` when the chosen action does not support it
   - Validates against the original request (action legality, payload conformance, extra bounds)
5. If parsing fails and correction retry is enabled, a bounded correction prompt is sent for one retry attempt.

### Baseline path

For baseline adapters (`random`, `block_always`, `greedy_damage`, `scripted_safe`):

1. The adapter selects an action deterministically from `legal_actions` using seeded RNG derived from `trace_seed + match_id + turn_id + state_hash + legal_actions_hash`.
2. Payload parameters are resolved from `payload_spec` constraints.
3. No prompt rendering or API call occurs.

### Fallback path

If the provider call times out, disconnects, or returns unparseable/illegal output:

1. `orchestrator.py` catches the failure and delegates to `fallback.py:build_fallback_decision()`.
2. The fallback strategy depends on `fallback_mode`:
   - **`safe_continue`** — picks a defensive/low-commitment action from the legal set
   - **`heuristic_guard`** — picks the best defensive action using keyword scoring
   - **`last_valid_replayable`** — replays the last valid decision if still legal, else falls back to `safe_continue`
3. The resulting `ActionDecision` is tagged with a `fallback_reason` enum.

### Timeout enforcement

`orchestrator.py` wraps the provider call in `asyncio.timeout(deadline_ms / 1000)`. The timeout is either `request.deadline_ms` or the configured `decision_timeout_ms`, whichever applies.

## Phase 7: Artifact Persistence (Daemon)

After decision resolution, `resolve_adapter_decision()` writes to the `MatchArtifactWriter`:

- **`prompts.jsonl`** — rendered prompt text, request payload, policy ID, prompt version (when `logging.prompts` is enabled)
- **`decisions.jsonl`** — request/decision pairs for every turn

`server.py` also writes telemetry events:
- `TurnRequested` before adapter dispatch
- `DecisionReceived` or `DecisionFallback` after resolution

All writes are `fsync`'d and JSONL-formatted.

## Phase 8: Response Transport (Daemon → Mod)

The `ActionDecision` is wrapped in a v2 envelope and sent back over the WebSocket:

```json
{
  "type": "action_decision",
  "version": "v2",
  "ts": "2026-03-12T00:00:01Z",
  "payload": {
    "match_id": "...",
    "turn_id": 1,
    "action": "block",
    "data": null,
    "extra": { "di": null, "feint": false, "reverse": false, "prediction": null }
  }
}
```

## Phase 9: Validation And Application (Mod)

`DecisionValidator.gd` checks the response before application:

- `match_id` and `turn_id` match the current turn (rejects stale responses)
- `action` is in the current legal set
- `data` conforms to the action's `payload_spec`
- `extra` fields (DI bounds, feint/reverse availability) are legal

If validation fails, `FallbackHandler.gd` selects a safe local fallback.

`ActionApplier.gd` applies the validated decision:

- Primary path: calls the native `on_action_selected(action, data, extra)` on the fighter object
- Compatibility fallback: writes directly to `queued_action`, `queued_data`, `queued_extra` fields

## Phase 10: Match End

When the game emits `game_ended` followed by `game_won(winner)`:

1. `TurnHook.gd` builds a `match_ended` envelope with `winner`, `end_reason`, `total_turns`, `end_tick`, `end_frame`, optional `replay_path`, and `errors`.
2. The envelope is sent to the daemon.
3. `server.py` receives it, breaks the match loop, and calls `writer.finalize()`.
4. The writer produces `result.json`, `metrics.json`, and `replay_index.json` in the run directory.

## Phase 11: Replay Recording (Optional)

After the match ends, the game automatically replays the match ~120 ticks later — playing back stored inputs without decision-time pauses. This produces a smooth, real-time-looking playback.

1. `TurnHook.gd` calls `ReplayManager.save_replay()` to save the replay file (the game only autosaves for multiplayer, but our mod forces a save for singleplayer matches).
2. `TurnHook.gd` sets `Global.playback_speed_mod = 1` and monitors for the auto-replay.
3. When `Global.current_game` changes to a new instance and `ReplayManager.playback` is `true`, the mod sends a `ReplayStarted` event with the display number.
4. The daemon receives the event and starts `ffmpeg -f x11grab` on the Xvfb display (via `orb run` in the VM).
5. When the replay game's `game_finished` becomes `true`, the mod sends `ReplayEnded`.
6. The daemon stops ffmpeg, pulls the video and replay file from the VM into the run directory.

Replay video recording is enabled by default. It can be controlled via `replay_capture.enabled` in the daemon config file, or overridden with `--record-replay` / `--no-record-replay` CLI flags. The mod-side replay saving always occurs regardless of this setting, and the daemon now still attempts to pull `match.replay` even when video capture is disabled.

## Artifact Layout After A Complete Match

```
runs/<timestamp>_<match_id>/
  manifest.json       ← config, versions, seed (written at match start)
  events.jsonl        ← lifecycle events
  decisions.jsonl     ← per-turn request/decision pairs
  prompts.jsonl       ← prompt traces (when logging.prompts enabled)
  metrics.json        ← latency stats, fallback rate, token usage
  result.json         ← winner, end reason, turn count, status
  replay_index.json   ← per-turn pointers into decisions and prompts
  stderr.log          ← error output
  replay.mp4          ← replay video (when replay_capture.enabled, default on)
  match.replay        ← game replay file (always attempted; normally present even when replay video is disabled)
```
