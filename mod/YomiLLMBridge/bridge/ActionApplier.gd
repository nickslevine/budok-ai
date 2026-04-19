extends Reference

# Injects validated decisions into the game's native decision pipeline.
# Prefer the fighter's native action-selection hook when available so ready-state
# and downstream commit logic stay aligned with the game's own lifecycle.

const NATIVE_APPLY_METHOD := "on_action_selected"
const ACTION_SELECTED_SIGNAL := "action_selected"


func apply_decision(decision_payload: Dictionary, fighter) -> Dictionary:
	# Returns a result dict with "applied" bool and optional "error" string.
	if fighter == null:
		return {"applied": false, "error": "fighter is null"}

	var action_name = str(decision_payload.get("action", ""))
	if action_name == "":
		return {"applied": false, "error": "empty action name"}

	var queued_data = _resolve_queued_data(decision_payload.get("data"))
	var queued_extra = _resolve_queued_extra(decision_payload.get("extra"))

	if fighter.has_method(NATIVE_APPLY_METHOD):
		fighter.call(NATIVE_APPLY_METHOD, action_name, queued_data, queued_extra)
		return _build_apply_result(action_name, fighter, "native_method")

	if fighter.has_signal(ACTION_SELECTED_SIGNAL):
		fighter.emit_signal(ACTION_SELECTED_SIGNAL, action_name, queued_data, queued_extra)
		return _build_apply_result(action_name, fighter, "signal_emit")

	if not _can_write_queued_fields(fighter):
		return {"applied": false, "error": "fighter is missing native apply hooks"}

	# Retain queued-field mutation only as a compatibility fallback for harnesses.
	fighter.queued_action = action_name
	fighter.queued_data = queued_data
	fighter.queued_extra = queued_extra

	return _build_apply_result(action_name, fighter, "queued_fields")


func _resolve_queued_data(data):
	# Pass through dict or null; the engine accepts both.
	# Single-child ActionUIData nodes return flat data (e.g. {"x": 70, "y": -70})
	# from get_data(), but the daemon wraps them in the child name
	# (e.g. {"Aim": {"x": 70, "y": -70}}).  The game's state code accesses
	# data.x / data.y directly, so we must unwrap single-key dicts where the
	# value is itself a dict (or the FixedMath library panics on null access).
	#
	# Also normalize whole-number TYPE_REAL values back to TYPE_INT. Godot 3.5.1
	# JSON.parse() turns all numbers into floats, but the game's action/state code
	# often expects integer payloads (counts, slider values, XY percentages).
	if data == null:
		return null
	if data is Dictionary:
		var result = data.duplicate(true)
		if result.size() == 1:
			var only_key = result.keys()[0]
			var only_value = result[only_key]
			if only_value is Dictionary:
				return _normalize_game_value(only_value)
		return _normalize_game_value(result)
	return null


func _normalize_game_value(value):
	if value is Dictionary:
		var normalized = {}
		for key in value.keys():
			normalized[key] = _normalize_game_value(value[key])
		return normalized
	if value is Array:
		var normalized_array = []
		for item in value:
			normalized_array.append(_normalize_game_value(item))
		return normalized_array
	if typeof(value) == TYPE_REAL and value == floor(value):
		return int(value)
	return value


func _resolve_queued_extra(extra):
	# Normalize the extra dict into the shape the engine expects.
	# The game's process_extra() checks for uppercase "DI", and
	# prediction is an integer (-1 = none) not a dict.
	if extra == null:
		return null
	if not (extra is Dictionary):
		return null

	var resolved = {}

	# The game checks "DI" (uppercase) in process_extra(); lowercase "di" is ignored.
	# IMPORTANT: The game's fixed-point math library expects string representations
	# of numbers (e.g. "50", "-100"), NOT raw integers.  BaseChar.current_di stores
	# {"x": "0", "y": "0"} and xy_to_dir / vec_mul parse these strings.  Passing
	# raw ints causes Rust panics ("invalid digit found in string") which accumulate
	# and eventually crash the game.
	var di = extra.get("di", extra.get("DI"))
	if di != null and di is Dictionary:
		resolved["DI"] = {
			"x": str(int(di.get("x", 0))),
			"y": str(int(di.get("y", 0))),
		}

	resolved["feint"] = bool(extra.get("feint", false))
	resolved["reverse"] = bool(extra.get("reverse", false))

	# The game's get_extra() sets prediction to an integer category index
	# (-1 = no prediction). Null or dict values are not expected.
	var prediction = extra.get("prediction")
	if prediction == null or prediction is Dictionary:
		resolved["prediction"] = -1
	else:
		resolved["prediction"] = int(prediction)

	return resolved


func _build_apply_result(action_name: String, fighter, apply_path: String) -> Dictionary:
	return {
		"applied": true,
		"action": action_name,
		"apply_path": apply_path,
		"queued_action": _maybe_get_property(fighter, "queued_action"),
		"queued_data": _duplicate_value(_maybe_get_property(fighter, "queued_data")),
		"queued_extra": _duplicate_value(_maybe_get_property(fighter, "queued_extra")),
	}


func _can_write_queued_fields(fighter) -> bool:
	return (
		_has_property(fighter, "queued_action")
		and _has_property(fighter, "queued_data")
		and _has_property(fighter, "queued_extra")
	)


func _maybe_get_property(target, property_name: String):
	if not _has_property(target, property_name):
		return null
	return target.get(property_name)


func _has_property(target, property_name: String) -> bool:
	if target == null:
		return false
	for property_info in target.get_property_list():
		if str(property_info.get("name", "")) == property_name:
			return true
	return false


func _duplicate_value(value):
	if value is Dictionary or value is Array:
		return value.duplicate(true)
	return value
