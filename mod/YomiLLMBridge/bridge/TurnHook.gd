extends Node

signal status_changed(snapshot)

# Detects actionable turns, emits DecisionRequest envelopes to the daemon,
# receives and validates responses, applies decisions or triggers fallback,
# and emits telemetry events throughout the lifecycle.
# Extends Node (not RefCounted) because it needs _process() and scene tree access.

var _bridge_client = null
var _config = {}
var _game = null
var _game_connected = false
var _match_id = ""
var _turn_id = 0
var _observation_builder = null
var _legal_action_builder = null
var _decision_validator = null
var _action_applier = null
var _fallback_handler = null
var _telemetry = null
var _protocol_codec = null
var _identifier_factory = null
var _runtime_compatibility = null
var _status_snapshot = {
	"state": "waiting_for_game",
	"compatible": null,
	"match_id": "",
	"errors": [],
	"details": {},
	"match_ended": null,
}
var _compatibility_signature = ""
var _match_end_pending = {}
var _match_ended = false
var _replay_snapshot = {}
var _replay_saved = false
var _awaiting_replay = false
var _replay_recording = false
var _replay_wait_started_ms = 0
var _replay_end_grace_ms = 0
var _original_game_id = 0
var _replay_game = null

# Pending requests keyed by "p1_turnid" or "p2_turnid" for correlation
var _pending_requests = {}
# Timestamps (OS.get_ticks_msec) for pending requests, same keys
var _pending_timestamps = {}
# Per-match turn history for context in observations
var _turn_history = []
# Staging area: tracks decisions per turn_id so we can build history entries
# once both players have acted. Keyed by turn_id -> { "p1_action": ..., ... }
var _turn_decisions = {}


func attach(bridge_client, config: Dictionary) -> void:
	_bridge_client = bridge_client
	_config = config
	_observation_builder = load(_get_script_dir() + "/ObservationBuilder.gd").new()
	_legal_action_builder = load(_get_script_dir() + "/LegalActionBuilder.gd").new()
	_decision_validator = load(_get_script_dir() + "/DecisionValidator.gd").new()
	_action_applier = load(_get_script_dir() + "/ActionApplier.gd").new()
	_fallback_handler = load(_get_script_dir() + "/FallbackHandler.gd").new()
	_telemetry = load(_get_script_dir() + "/Telemetry.gd").new()
	_protocol_codec = load(_get_script_dir() + "/ProtocolCodec.gd").new()
	_identifier_factory = load(_get_script_dir() + "/IdentifierFactory.gd").new()
	_runtime_compatibility = load(_get_script_dir() + "/RuntimeCompatibility.gd").new()

	# Subscribe to daemon messages and disconnect events
	_bridge_client.connect("daemon_message", self, "_on_daemon_message")
	_bridge_client.connect("daemon_disconnected", self, "_on_daemon_disconnected")

	set_process(true)
	_publish_status("waiting_for_game", null, [], {})


func _process(_delta: float) -> void:
	# Post-match: monitor for replay lifecycle
	if _match_ended:
		_monitor_replay_lifecycle()
		return
	if _game_connected:
		_check_timeouts()
		return
	if not _has_global_game():
		_publish_status("waiting_for_game", null, [], {})
		return

	var compatibility = _runtime_compatibility.check(
		Global.current_game,
		get_tree().get_root(),
		_config
	)
	if not bool(compatibility.get("compatible", false)):
		_publish_compatibility_failure(compatibility)
		return
	_attach_to_game(Global.current_game, compatibility)


func _check_already_actionable() -> void:
	if _game == null or _match_ended:
		return
	var actionable = _get_actionable_players()
	if not actionable.empty():
		print("YomiLLMBridge: players already actionable on attach: %s" % [actionable])
		_on_player_actionable()


func _on_player_actionable() -> void:
	if _game == null or _match_ended:
		return

	# Retroactively enrich the most recent history entry with outcome feedback
	# At this point the game has resolved the previous tick, so HP/position
	# reflect post-resolution state
	_enrich_last_history_entry()

	var actionable_players = _get_actionable_players()
	for player_id in actionable_players:
		# Guard: skip if we already have a pending request for this player.
		# The game fires player_actionable multiple times per tick; without
		# this check each signal causes a separate LLM API call for the same
		# game state, wasting tokens and time.
		if _has_pending_for_player(player_id):
			continue

		var fighter = _game.p1 if player_id == "p1" else _game.p2
		var action_buttons = _find_action_buttons(player_id)
		if action_buttons == null:
			printerr("YomiLLMBridge TurnHook could not find action buttons for %s" % player_id)
			continue

		_turn_id += 1
		var observation = _observation_builder.build_observation(_game, fighter, _turn_history)
		var legal_actions = _legal_action_builder.build_legal_actions(_game, fighter, action_buttons)

		var state_hash = _protocol_codec.sha256_hex(observation)
		var legal_actions_hash = _protocol_codec.sha256_hex(legal_actions)

		var decision_request_payload = {
			"match_id": _match_id,
			"turn_id": _turn_id,
			"player_id": player_id,
			"deadline_ms": int(_config.get("decision_timeout_ms", 2500)),
			"state_hash": state_hash,
			"legal_actions_hash": legal_actions_hash,
			"decision_type": "turn_action",
			"observation": observation,
			"legal_actions": legal_actions,
		}
		var decision_request_envelope = _protocol_codec.build_envelope(
			"decision_request",
			decision_request_payload
		)

		# Track pending request for correlation
		var pending_key = _pending_key(player_id, _turn_id)
		_pending_requests[pending_key] = decision_request_payload
		_pending_timestamps[pending_key] = OS.get_ticks_msec()

		_bridge_client._send_json_message(decision_request_envelope)

		_telemetry.emit_turn_requested(
			_turn_id, player_id, state_hash, legal_actions_hash,
			OS.get_ticks_msec()
		)


func _on_daemon_message(envelope: Dictionary) -> void:
	if _match_ended:
		return
	if str(envelope.get("type", "")) != "action_decision":
		return
	if not envelope.has("payload") or not (envelope["payload"] is Dictionary):
		return
	var payload: Dictionary = envelope["payload"]

	var player_id = _find_player_for_decision(payload)
	if player_id == "":
		return

	var pending_key = _pending_key(player_id, int(payload.get("turn_id", -1)))
	var request = _pending_requests.get(pending_key)
	if request == null:
		# No pending request for this decision; could be stale
		return

	var sent_time = _pending_timestamps.get(pending_key, OS.get_ticks_msec())
	var latency_ms = OS.get_ticks_msec() - sent_time

	# Validate the decision against the pending request
	var validation_error = _decision_validator.validate_decision(envelope, request)

	if validation_error == "":
		# Valid decision: apply it
		_apply_valid_decision(payload, request, player_id, latency_ms)
	else:
		# Invalid decision: use fallback
		_apply_fallback(request, player_id, validation_error, latency_ms)


func _apply_valid_decision(
	payload: Dictionary,
	request: Dictionary,
	player_id: String,
	latency_ms: int
) -> void:
	var fighter = _game.p1 if player_id == "p1" else _game.p2

	var apply_result = _action_applier.apply_decision(payload, fighter)
	if not bool(apply_result.get("applied", false)):
		printerr(
			"YomiLLMBridge APPLY FAIL action=%s turn=%s player=%s error=%s"
			% [
				str(payload.get("action", "")),
				int(request.get("turn_id", 0)),
				player_id,
				str(apply_result.get("error", "unknown")),
			]
		)
		# Application failed, use fallback
		_apply_fallback(request, player_id, "malformed_output", latency_ms)
		return
	printerr(
		"YomiLLMBridge APPLY OK action=%s turn=%s player=%s path=%s"
		% [
			str(payload.get("action", "")),
			int(request.get("turn_id", 0)),
			player_id,
			str(apply_result.get("apply_path", "unknown")),
		]
	)

	var action_name = str(payload.get("action", ""))
	var policy_id = str(payload.get("policy_id", ""))

	# Record as last valid decision for potential replay
	_fallback_handler.record_valid_decision(player_id, payload)

	# Record decision into turn history (keyed by game tick for proper pairing)
	var game_tick = int(_game.current_tick) if _game != null else 0
	_record_turn_decision(int(request.get("turn_id", 0)), game_tick, player_id, action_name, false)

	_telemetry.emit_decision_received(
		int(request.get("turn_id", 0)), player_id, action_name, latency_ms, policy_id
	)
	_telemetry.emit_decision_applied(
		int(request.get("turn_id", 0)), player_id, action_name, latency_ms
	)

	# Clear pending request
	var pending_key = _pending_key(player_id, int(request.get("turn_id", 0)))
	_clear_pending(pending_key)


func _apply_fallback(
	request: Dictionary,
	player_id: String,
	fallback_reason: String,
	latency_ms: int
) -> void:
	var fallback_mode = str(_config.get("fallback_mode", "safe_continue"))
	var fallback_decision = _fallback_handler.choose_fallback(
		request, fallback_reason, fallback_mode
	)

	var fighter = _game.p1 if player_id == "p1" else _game.p2
	var apply_result = _action_applier.apply_decision(fallback_decision, fighter)

	var fallback_action = str(fallback_decision.get("action", ""))
	if not bool(apply_result.get("applied", false)):
		printerr(
			"YomiLLMBridge FALLBACK APPLY FAIL action=%s turn=%s player=%s error=%s"
			% [
				fallback_action,
				int(request.get("turn_id", 0)),
				player_id,
				str(apply_result.get("error", "unknown")),
			]
		)
	else:
		printerr(
			"YomiLLMBridge FALLBACK APPLY OK action=%s turn=%s player=%s path=%s"
			% [
				fallback_action,
				int(request.get("turn_id", 0)),
				player_id,
				str(apply_result.get("apply_path", "unknown")),
			]
		)
	var strategy = str(fallback_decision.get("policy_id", "fallback/safe_continue"))
	# Strip "fallback/" prefix for the strategy name
	if strategy.begins_with("fallback/"):
		strategy = strategy.substr(9)

	# Record fallback into turn history (keyed by game tick for proper pairing)
	var game_tick = int(_game.current_tick) if _game != null else 0
	_record_turn_decision(int(request.get("turn_id", 0)), game_tick, player_id, fallback_action, true)

	_telemetry.emit_decision_fallback(
		int(request.get("turn_id", 0)),
		player_id,
		fallback_reason,
		fallback_action,
		strategy,
		latency_ms
	)

	# Clear pending request
	var pending_key = _pending_key(player_id, int(request.get("turn_id", 0)))
	_clear_pending(pending_key)


func _record_turn_decision(turn_id: int, game_tick: int, player_id: String, action_name: String, was_fallback: bool) -> void:
	"""Record a player's decision for a turn. Both players' decisions in the
	same game tick are merged into a single history entry keyed by tick."""
	var tick_key = str(game_tick)
	if not _turn_decisions.has(tick_key):
		_turn_decisions[tick_key] = {"turn_id": turn_id, "game_tick": game_tick}
	var entry = _turn_decisions[tick_key]
	var prefix = player_id  # "p1" or "p2"
	entry[prefix + "_action"] = action_name
	entry[prefix + "_was_fallback"] = was_fallback

	# Snapshot current HP and positions for each recorded decision
	# (We always snapshot the latest state so it reflects post-decision state)
	if _game != null:
		var p1_pos = _game.p1.get_pos()
		var p2_pos = _game.p2.get_pos()
		entry["p1_hp"] = int(_game.p1.hp)
		entry["p2_hp"] = int(_game.p2.hp)
		entry["p1_pos"] = {"x": int(p1_pos.x), "y": int(p1_pos.y)}
		entry["p2_pos"] = {"x": int(p2_pos.x), "y": int(p2_pos.y)}

	# Build history entry once we have at least one player's decision
	# (In simultaneous turns both arrive; in single-player turns only one does)
	if entry.has("p1_action") or entry.has("p2_action"):
		var history_entry = {
			"turn_id": int(entry.get("turn_id", turn_id)),
			"game_tick": game_tick,
		}
		if entry.has("p1_action"):
			history_entry["p1_action"] = entry["p1_action"]
			history_entry["p1_was_fallback"] = entry.get("p1_was_fallback", false)
		if entry.has("p2_action"):
			history_entry["p2_action"] = entry["p2_action"]
			history_entry["p2_was_fallback"] = entry.get("p2_was_fallback", false)
		if entry.has("p1_hp"):
			history_entry["p1_hp"] = entry["p1_hp"]
			history_entry["p2_hp"] = entry["p2_hp"]
		if entry.has("p1_pos"):
			history_entry["p1_pos"] = entry["p1_pos"]
			history_entry["p2_pos"] = entry["p2_pos"]

		# Check if we already have this game_tick in history; update if so
		var found = false
		for i in range(_turn_history.size() - 1, -1, -1):
			if int(_turn_history[i].get("game_tick", -1)) == game_tick:
				_turn_history[i] = history_entry
				found = true
				break
		if not found:
			_turn_history.append(history_entry)
		# Keep history sorted by game_tick
		_turn_history.sort_custom(self, "_sort_history_by_tick")


func _on_daemon_disconnected(_details: Dictionary) -> void:
	if _match_ended:
		return
	# Resolve all pending requests as disconnected
	var keys_to_resolve = _pending_requests.keys().duplicate()
	for pending_key in keys_to_resolve:
		var request = _pending_requests.get(pending_key)
		if request == null:
			continue
		var player_id = str(request.get("player_id", ""))
		var sent_time = _pending_timestamps.get(pending_key, OS.get_ticks_msec())
		var latency_ms = OS.get_ticks_msec() - sent_time
		_apply_fallback(request, player_id, "disconnect", latency_ms)


func _check_timeouts() -> void:
	if _match_ended:
		return
	var timeout_ms = int(_config.get("decision_timeout_ms", 2500))
	var now = OS.get_ticks_msec()
	var keys_to_resolve = []

	for pending_key in _pending_timestamps:
		var sent_time = _pending_timestamps[pending_key]
		if now - sent_time >= timeout_ms:
			keys_to_resolve.append(pending_key)

	for pending_key in keys_to_resolve:
		var request = _pending_requests.get(pending_key)
		if request == null:
			continue
		var player_id = str(request.get("player_id", ""))
		var latency_ms = now - _pending_timestamps.get(pending_key, now)
		_apply_fallback(request, player_id, "timeout", latency_ms)


func _find_player_for_decision(payload: Dictionary) -> String:
	# Match decision to player via turn_id and match_id correlation
	var decision_match_id = str(payload.get("match_id", ""))
	var decision_turn_id = int(payload.get("turn_id", -1))

	for player_id in ["p1", "p2"]:
		var key = _pending_key(player_id, decision_turn_id)
		if _pending_requests.has(key):
			var request = _pending_requests[key]
			if str(request.get("match_id", "")) == decision_match_id:
				return player_id

	return ""


func _pending_key(player_id: String, turn_id: int) -> String:
	return "%s_%d" % [player_id, turn_id]


func _clear_pending(pending_key: String) -> void:
	_pending_requests.erase(pending_key)
	_pending_timestamps.erase(pending_key)


func _has_pending_for_player(player_id: String) -> bool:
	"""Return true if there is already an outstanding decision request for this player."""
	for key in _pending_requests:
		if str(key).begins_with(player_id + "_"):
			return true
	return false


func _get_actionable_players() -> Array:
	var players = []
	if _game.p1_turn and _is_fighter_interruptable(_game.p1):
		players.append("p1")
	if _game.p2_turn and _is_fighter_interruptable(_game.p2):
		players.append("p2")
	return players


func _is_fighter_interruptable(fighter) -> bool:
	if fighter == null:
		return false
	return bool(fighter.state_interruptable)


func _find_action_buttons(player_id: String):
	var node_name = "P1ActionButtons" if player_id == "p1" else "P2ActionButtons"
	return get_tree().get_root().find_node(node_name, true, false)


func _has_global_game() -> bool:
	if not ("current_game" in Global):
		return false
	return Global.current_game != null


func get_status() -> Dictionary:
	return _status_snapshot.duplicate(true)


func _attach_to_game(game, compatibility: Dictionary) -> void:
	if _game_connected and _game == game:
		return

	_game = game
	_match_id = _identifier_factory.new_match_id()
	_turn_id = 0
	_match_end_pending = {}
	_match_ended = false
	_compatibility_signature = ""
	_pending_requests.clear()
	_pending_timestamps.clear()
	_turn_history = []
	_turn_decisions = {}
	_replay_snapshot = _snapshot_replay_directory("user://replay/autosave")
	_game.connect("player_actionable", self, "_on_player_actionable")
	_game.connect("game_ended", self, "_on_game_ended")
	_game.connect("game_won", self, "_on_game_won")
	_game_connected = true

	# Apply match_options overrides before any turns fire
	_apply_match_options()

	# Configure telemetry now that we have a match_id
	_telemetry.configure(_bridge_client, _config, _match_id)

	_publish_status("active", true, [], compatibility.get("details", {}))
	print("YomiLLMBridge TurnHook attached to game, match_id=%s" % _match_id)

	# If players are already actionable (signal fired before we attached), trigger immediately
	call_deferred("_check_already_actionable")


func _apply_match_options() -> void:
	# Apply HP override via shared helper
	_apply_match_options_to_game(_game)

	# Scale the match timer to match the HP pool.
	# Game defaults: MAX_HEALTH=1500, game.time=3000 (2 ticks per HP).
	# Observed: baseline matches use ~1.9 ticks/HP, LLM matches use ~2.5 ticks/HP.
	# We use 3 ticks per HP for a comfortable margin that avoids timeouts.
	var match_options = _config.get("match_options", {})
	if not (match_options is Dictionary) or match_options.empty():
		return
	var starting_hp = match_options.get("starting_hp", null)
	var match_time = match_options.get("match_time", null)
	if match_time != null and int(match_time) > 0:
		_game.time = int(match_time)
		print("YomiLLMBridge applied match_time=%d ticks" % int(match_time))
	elif starting_hp != null and int(starting_hp) > 0:
		var auto_time = int(starting_hp) * 3
		_game.time = auto_time
		print("YomiLLMBridge auto-scaled match_time=%d ticks for starting_hp=%d" % [auto_time, int(starting_hp)])


func _apply_match_options_to_game(game) -> void:
	"""Apply match option overrides (HP, timer) to any game instance.
	Used for both the live match and the replay game."""
	if game == null:
		return
	var match_options = _config.get("match_options", {})
	if not (match_options is Dictionary) or match_options.empty():
		return

	var starting_hp = match_options.get("starting_hp", null)
	if starting_hp != null and int(starting_hp) > 0:
		var hp_value = int(starting_hp)
		for fighter in [game.p1, game.p2]:
			fighter.MAX_HEALTH = hp_value
			fighter.hp = hp_value
			fighter.trail_hp = hp_value
		# Update HUD health bar max values so bars render as full.
		# The HUD reads MAX_HEALTH during its own init (before our override),
		# so healthbar.max_value is still 1500. Fix it here.
		var hud = _find_hud(game)
		if hud != null:
			_sync_hud_health_bars(hud, hp_value)
		print("YomiLLMBridge applied starting_hp=%d to game instance" % hp_value)


func _find_hud(game) -> Node:
	"""Find the HudLayer node in the scene tree."""
	var root = get_tree().get_root() if get_tree() != null else null
	if root == null:
		return null
	return root.find_node("HudLayer", true, false)


func _sync_hud_health_bars(hud, max_hp: int) -> void:
	"""Update all health bar max_value and trail values to match the HP override."""
	for bar_name in [
		"P1HealthBar", "P2HealthBar",
		"P1HealthBarTrail", "P2HealthBarTrail",
		"P1GhostHealthBar", "P2GhostHealthBar",
		"P1GhostHealthBarTrail", "P2GhostHealthBarTrail",
	]:
		var bar = hud.find_node(bar_name, true, false)
		if bar != null:
			bar.max_value = max_hp
			bar.value = max_hp


func _publish_compatibility_failure(compatibility: Dictionary) -> void:
	var errors = compatibility.get("errors", [])
	var details = compatibility.get("details", {})
	var signature = to_json({"errors": errors, "details": details})
	_publish_status("compatibility_failed", false, errors, details)
	if signature == _compatibility_signature:
		return
	_compatibility_signature = signature
	for error_message in errors:
		printerr("YomiLLMBridge compatibility check failed: %s" % error_message)


func _publish_status(
	state: String,
	compatible,
	errors: Array,
	details: Dictionary,
	match_ended_payload = null
) -> void:
	var next_snapshot = {
		"state": state,
		"compatible": compatible,
		"match_id": _match_id,
		"errors": errors.duplicate(true),
		"details": details.duplicate(true),
		"match_ended": (
			match_ended_payload.duplicate(true)
			if match_ended_payload is Dictionary
			else match_ended_payload
		),
	}
	if to_json(next_snapshot) == to_json(_status_snapshot):
		return
	_status_snapshot = next_snapshot
	emit_signal("status_changed", get_status())


func _on_game_ended() -> void:
	if _match_ended:
		return
	_match_end_pending = _build_match_ended_payload(null)


func _on_game_won(winner) -> void:
	if _match_ended:
		return
	var payload = (
		_match_end_pending.duplicate(true)
		if not _match_end_pending.empty()
		else _build_match_ended_payload(winner)
	)
	payload["winner"] = _normalize_winner(winner)
	_finalize_match(payload)


func _finalize_match(payload: Dictionary) -> void:
	if _match_ended:
		return
	_match_ended = true
	_match_end_pending = {}
	_pending_requests.clear()
	_pending_timestamps.clear()
	_telemetry.emit_match_ended(payload)
	_publish_status("match_ended", true, [], _status_snapshot.get("details", {}), payload)
	print("YomiLLMBridge match ended: %s" % payload)

	# Save replay and begin monitoring for auto-replay playback
	_save_replay()
	_begin_replay_monitoring()


func _build_match_ended_payload(winner_override) -> Dictionary:
	var payload = {
		"match_id": _match_id,
		"winner": _normalize_winner(winner_override),
		"end_reason": _derive_end_reason(),
		"total_turns": _turn_id,
		"end_tick": _derive_end_tick(),
		"end_frame": _derive_end_frame(),
		"errors": [],
	}
	var replay_path = _find_replay_path()
	if replay_path != null:
		payload["replay_path"] = replay_path
	return payload


func _normalize_winner(winner_override):
	if winner_override == 1 or str(winner_override).to_lower() == "p1":
		return "p1"
	if winner_override == 2 or str(winner_override).to_lower() == "p2":
		return "p2"
	if _game == null:
		return null
	var p1_hp = int(_game.p1.hp)
	var p2_hp = int(_game.p2.hp)
	if p1_hp > p2_hp:
		return "p1"
	if p2_hp > p1_hp:
		return "p2"
	return null


func _derive_end_reason() -> String:
	if _game == null:
		return "resolved"
	if int(_game.current_tick) > int(_game.time):
		return "timeout"
	if int(_game.p1.hp) <= 0 or int(_game.p2.hp) <= 0:
		return "ko"
	return "resolved"


func _derive_end_tick() -> int:
	if _game == null:
		return 0
	if _has_property(_game, "game_end_tick"):
		return int(_game.game_end_tick)
	return int(_game.current_tick)


func _derive_end_frame() -> int:
	if _game == null:
		return 0
	return int(_game.current_tick)


func _save_replay() -> void:
	if _replay_saved or _game == null:
		return
	if not _game.has_method("get") and not ("match_data" in _game):
		printerr("YomiLLMBridge cannot save replay: game has no match_data")
		return

	var match_data = {}
	if "match_data" in _game:
		match_data = _game.match_data.duplicate(true) if _game.match_data is Dictionary else {}

	var timestamp = OS.get_unix_time()
	var file_name = "llm_%s_%d" % [_match_id.substr(0, 8), timestamp]
	var saved_file_name = ReplayManager.save_replay(match_data, file_name, true)
	_replay_saved = true

	var replay_path = _resolve_saved_replay_path(saved_file_name)
	if replay_path == null:
		replay_path = _find_replay_path()
	if replay_path != null:
		print("YomiLLMBridge replay saved: %s" % replay_path)
		_telemetry.emit_replay_saved(replay_path)
	else:
		var autosave_dir = ProjectSettings.globalize_path("user://replay/autosave")
		print(
			"YomiLLMBridge replay save called but file not found (saved_file_name=%s, autosave_dir=%s)"
			% [str(saved_file_name), autosave_dir]
		)


func _begin_replay_monitoring() -> void:
	if _game == null:
		return
	_original_game_id = _game.get_instance_id()
	_awaiting_replay = true
	_replay_recording = false
	_replay_game = null
	_replay_wait_started_ms = OS.get_ticks_msec()
	_replay_end_grace_ms = 0
	# Slow down replay playback for watchable recording.
	# playback_speed_mod controls tick processing: real_tick % mod == 0
	# mod=1: 60 ticks/s (full speed), mod=2: 30 ticks/s, mod=4: 15 ticks/s
	# We use mod=2 for half-speed: smooth enough while being watchable.
	Global.playback_speed_mod = 2

	# Send ReplayStarted immediately so the daemon starts ffmpeg BEFORE the
	# replay game is created. The game auto-starts replay ~120 ticks (~2s)
	# after match end. By sending the event now, ffmpeg has ~2-5 seconds to
	# start before the replay plays.
	var display = OS.get_environment("DISPLAY")
	if display == "":
		display = ":99"
	print("YomiLLMBridge starting replay recording on display %s" % display)
	_telemetry.emit_replay_started(display)


func _monitor_replay_lifecycle() -> void:
	if _awaiting_replay:
		# Check timeout (60 seconds to wait for auto-replay)
		if OS.get_ticks_msec() - _replay_wait_started_ms > 60000:
			print("YomiLLMBridge replay wait timed out")
			_awaiting_replay = false
			_telemetry.emit_replay_ended()
			return

		# Detect the replay game instance for end-of-replay monitoring.
		if not _has_global_game():
			return
		if Global.current_game == null:
			return
		if Global.current_game.get_instance_id() == _original_game_id:
			return

		# Found the replay game instance
		_awaiting_replay = false
		_replay_recording = true
		_replay_game = Global.current_game
		print("YomiLLMBridge replay game instance detected")

		# Apply HP override to the replay game so damage matches the live match.
		# The game creates replay fighters with default MAX_HEALTH=1500, but
		# the live match may have used a different starting_hp (e.g. 750).
		# Without this, the replay diverges: same attacks deal the same raw
		# damage but against 2x HP, so the KO never happens and the replay
		# freezes when it runs out of recorded actions.
		_apply_match_options_to_game(_replay_game)

	elif _replay_recording:
		# Monitor for replay end
		if _replay_game == null or not is_instance_valid(_replay_game):
			print("YomiLLMBridge replay game freed, recording complete")
			_replay_recording = false
			_telemetry.emit_replay_ended()
			_prevent_replay_loop()
			return
		if _replay_game.game_finished:
			# Add a grace period after game_finished to capture the post-KO
			# animation (slow-mo, victory text, etc.) before stopping recording.
			if _replay_end_grace_ms == 0:
				_replay_end_grace_ms = OS.get_ticks_msec()
				print("YomiLLMBridge replay game finished, waiting 5s for post-KO animation...")
				_prevent_replay_loop()
			elif OS.get_ticks_msec() - _replay_end_grace_ms > 5000:
				print("YomiLLMBridge replay recording complete (after grace period)")
				_replay_recording = false
				_telemetry.emit_replay_ended()


func _prevent_replay_loop() -> void:
	"""Break the game's infinite replay loop after we've captured one replay.

	game.gd _physics_process auto-calls start_playback() 120 ticks after
	game_finished, which creates a new replay game via main._on_playback_requested().
	Setting play_full=false prevents the loop from continuing."""
	if "play_full" in ReplayManager:
		ReplayManager.play_full = false
		print("YomiLLMBridge set ReplayManager.play_full=false to prevent replay loop")
	# Also try to stop the replay game from triggering another cycle
	if _replay_game != null and is_instance_valid(_replay_game):
		_replay_game.game_started = false


func _resolve_saved_replay_path(saved_file_name):
	if typeof(saved_file_name) != TYPE_STRING:
		return null
	var replay_file_name = str(saved_file_name)
	if replay_file_name == "":
		return null
	var replay_user_path = "user://replay/autosave/" + replay_file_name
	var replay_global_path = ProjectSettings.globalize_path(replay_user_path)
	var file = File.new()
	if file.file_exists(replay_user_path):
		return replay_global_path
	if file.file_exists(replay_global_path):
		return replay_global_path
	return null


func _find_replay_path():
	var after = _snapshot_replay_directory("user://replay/autosave")
	var best_path = ""
	var best_time = -1
	for path in after:
		var modified_at = int(after[path])
		if not str(path).ends_with(".replay"):
			continue
		if modified_at <= int(_replay_snapshot.get(path, -1)):
			continue
		if modified_at > best_time:
			best_time = modified_at
			best_path = str(path)
	if best_path != "":
		return ProjectSettings.globalize_path(best_path)

	for path in after:
		var modified_at = int(after[path])
		if not str(path).ends_with(".replay"):
			continue
		if modified_at > best_time:
			best_time = modified_at
			best_path = str(path)
	if best_path == "":
		return null
	return ProjectSettings.globalize_path(best_path)


func _snapshot_replay_directory(path: String) -> Dictionary:
	var snapshot = {}
	var directory = Directory.new()
	if directory.open(path) != OK:
		return snapshot
	directory.list_dir_begin(true, true)
	while true:
		var file_name = directory.get_next()
		if file_name == "":
			break
		if directory.current_is_dir():
			continue
		if not file_name.ends_with(".replay"):
			continue
		var file_path = path.plus_file(file_name)
		snapshot[file_path] = directory.get_modified_time(file_path)
	directory.list_dir_end()
	return snapshot


func _has_property(target, property_name: String) -> bool:
	if target == null or not target.has_method("get_property_list"):
		return false
	for property_data in target.get_property_list():
		if str(property_data.get("name", "")) == property_name:
			return true
	return false


func _enrich_last_history_entry() -> void:
	"""After the game resolves a tick, retroactively add outcome feedback to the
	most recent history entry by comparing post-resolution HP to the snapshot."""
	if _turn_history.empty() or _game == null:
		return
	var last = _turn_history[_turn_history.size() - 1]
	# Skip if already enriched
	if last.has("p1_hp_delta"):
		return

	var prev_p1_hp = int(last.get("p1_hp", int(_game.p1.hp)))
	var prev_p2_hp = int(last.get("p2_hp", int(_game.p2.hp)))
	var now_p1_hp = int(_game.p1.hp)
	var now_p2_hp = int(_game.p2.hp)

	var p1_delta = now_p1_hp - prev_p1_hp  # negative = took damage
	var p2_delta = now_p2_hp - prev_p2_hp

	last["p1_hp_delta"] = p1_delta
	last["p2_hp_delta"] = p2_delta

	# Update HP/position to post-resolution values
	last["p1_hp"] = now_p1_hp
	last["p2_hp"] = now_p2_hp
	var p1_pos = _game.p1.get_pos()
	var p2_pos = _game.p2.get_pos()
	last["p1_pos"] = {"x": int(p1_pos.x), "y": int(p1_pos.y)}
	last["p2_pos"] = {"x": int(p2_pos.x), "y": int(p2_pos.y)}

	# Infer outcomes from HP deltas and game state
	last["p1_outcome"] = _infer_outcome(p1_delta, p2_delta, _game.p1, _game.p2, last)
	last["p2_outcome"] = _infer_outcome(p2_delta, p1_delta, _game.p2, _game.p1, last)


func _infer_outcome(my_delta: int, opp_delta: int, my_fighter, opp_fighter, entry: Dictionary) -> String:
	"""Infer what happened to a player based on HP deltas and state."""
	# I took damage
	if my_delta < 0:
		if int(opp_fighter.combo_count) > 1:
			return "combo"
		if entry.has("p2_action") and entry.has("p1_action"):
			# Opponent had a grab action
			var opp_prefix = "p2_action" if my_fighter == _game.p1 else "p1_action"
			var opp_action = str(entry.get(opp_prefix, ""))
			if opp_action.to_lower().find("grab") != -1 or opp_action == "Lasso" or opp_action == "IzunaDrop" or opp_action == "Impale":
				return "grabbed"
		return "hit"
	# I dealt damage (opponent took damage, I didn't)
	if opp_delta < 0 and my_delta >= 0:
		return "hit"
	# Both took no damage
	if my_delta == 0 and opp_delta == 0:
		if int(my_fighter.blockstun_ticks) > 0:
			return "blocked"
		var my_action_key = "p1_action" if my_fighter == _game.p1 else "p2_action"
		var my_action = str(entry.get(my_action_key, ""))
		var opp_action_key = "p2_action" if my_fighter == _game.p1 else "p1_action"
		var opp_action = str(entry.get(opp_action_key, ""))
		# Both attacked but no damage = clashed
		if _is_attack_action(my_action) and _is_attack_action(opp_action):
			return "clashed"
		if _is_defense_action(my_action):
			return "blocked"
		return "neutral"
	return "neutral"


func _is_attack_action(action_name: String) -> bool:
	"""Heuristic: common attack action patterns."""
	var lower = action_name.to_lower()
	return (
		lower.find("slash") != -1 or lower.find("kick") != -1 or
		lower.find("punch") != -1 or lower.find("combo") != -1 or
		lower.find("swipe") != -1 or lower.find("stinger") != -1 or
		lower.find("shoot") != -1 or lower.find("blast") != -1 or
		lower.find("cleave") != -1 or lower.find("strike") != -1 or
		lower.find("slice") != -1 or lower.find("bolt") != -1 or
		lower == "grab" or lower == "lasso" or lower.find("impale") != -1
	)


func _is_defense_action(action_name: String) -> bool:
	"""Heuristic: common defense action patterns."""
	var lower = action_name.to_lower()
	return (
		lower.find("parry") != -1 or lower.find("block") != -1 or
		lower.find("roll") != -1 or lower.find("dodge") != -1 or
		lower.find("burst") != -1 or lower == "wait" or
		lower == "spotdodge" or lower == "techroll"
	)


func _sort_history_by_tick(a: Dictionary, b: Dictionary) -> bool:
	return int(a.get("game_tick", 0)) < int(b.get("game_tick", 0))


func _get_script_dir() -> String:
	return get_script().resource_path.get_base_dir()
