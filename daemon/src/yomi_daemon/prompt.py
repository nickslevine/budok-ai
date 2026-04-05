"""Deterministic prompt rendering utilities for provider-backed policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from yomi_daemon.protocol import DecisionRequest, JsonObject, default_prediction_spec
from yomi_daemon.validation import REPO_ROOT


PROMPTS_DIR = REPO_ROOT / "prompts"
MOVE_CATALOG_PATH = PROMPTS_DIR / "move_catalog.json"
CHARACTER_SELECT_TEMPLATE = PROMPTS_DIR / "character_select_v1.md"
DEFAULT_PROMPT_VERSION = "minimal_v1"
VALID_CHARACTERS = ("Ninja", "Cowboy", "Wizard", "Robot", "Mutant")
PROMPT_VERSION_ALIASES = {
    "reasoning_enabled_v1": "reasoning_v1",
}

_move_catalog_cache: JsonObject | None = None


class PromptTemplateError(ValueError):
    """Raised when a configured prompt template cannot be rendered."""


class PromptTemplateVariant(StrEnum):
    MINIMAL = "minimal"
    STRATEGIC = "strategic"
    FEW_SHOT = "few_shot"
    REASONING_ENABLED = "reasoning_enabled"


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    prompt_version: str
    variant: PromptTemplateVariant
    prompt_text: str
    template_path: Path


def resolve_prompt_version(
    request: DecisionRequest,
    *,
    configured_prompt_version: str | None = None,
) -> str:
    selected = request.prompt_version or configured_prompt_version or DEFAULT_PROMPT_VERSION
    return PROMPT_VERSION_ALIASES.get(selected, selected)


def render_prompt(
    request: DecisionRequest,
    *,
    configured_prompt_version: str | None = None,
    policy_id: str | None = None,
) -> RenderedPrompt:
    prompt_version = resolve_prompt_version(
        request,
        configured_prompt_version=configured_prompt_version,
    )
    template_path = PROMPTS_DIR / f"{prompt_version}.md"
    if not template_path.is_file():
        raise PromptTemplateError(
            f"prompt template {prompt_version!r} was not found at {template_path}"
        )

    variant = _variant_for_prompt_version(prompt_version)
    cheat_sheet = _tactical_cheat_sheet(request)
    char_guide = _character_guide(request)
    sections = [
        template_path.read_text(encoding="utf-8").strip(),
        _compact_output_contract(),
        f"## Turn Context\n```json\n{_json_dump(_turn_context(request, policy_id=policy_id))}\n```",
        _situation_summary(request),
        *([cheat_sheet] if cheat_sheet else []),
        *([char_guide] if char_guide else []),
        f"## Observation\n```json\n{_json_dump(_compact_observation(request))}\n```",
        _grouped_legal_actions_section(request),
    ]

    return RenderedPrompt(
        prompt_version=prompt_version,
        variant=variant,
        prompt_text="\n\n".join(sections).strip() + "\n",
        template_path=template_path,
    )


def _compact_output_contract() -> str:
    """Concise output contract replacing the full JSON schema to save ~1200 tokens."""
    return (
        "## Output Contract\n"
        "Return exactly one JSON object (no Markdown fences). Required field: `action` (string "
        "from the legal actions list). Optional fields: `data` (object, only when action has "
        "payload_spec), `extra` (object with `di: {x,y}` integers -100..100, `feint`: bool, "
        "`reverse`: bool — only include if the action supports them), `reasoning` (string), "
        "`notes` (string).\n"
        'Example: `{"action": "HSlash2", "reasoning": "opponent is blocking, switching to grab next"}`\n'
        'Example with payload: `{"action": "ParryHigh", "data": {"Melee Parry Timing": 10}}`'
    )


def decision_output_json_schema() -> JsonObject:
    """Full schema kept for validation/reference use, no longer rendered in prompts."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "description": "Exact legal action identifier chosen for this turn.",
            },
            "data": {
                "type": "object",
                "description": (
                    "Optional action payload object. Only include keys allowed by the chosen "
                    "legal action's payload_spec."
                ),
                "additionalProperties": True,
            },
            "extra": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "di": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["x", "y"],
                        "properties": {
                            "x": {"type": "integer", "minimum": -100, "maximum": 100},
                            "y": {"type": "integer", "minimum": -100, "maximum": 100},
                        },
                    },
                    "feint": {"type": "boolean"},
                    "reverse": {"type": "boolean"},
                    "prediction": {
                        "anyOf": [
                            default_prediction_spec(),
                            {"type": "null"},
                        ]
                    },
                },
            },
            "notes": {
                "type": "string",
                "description": "Optional brief note about the tactical choice.",
            },
            "reasoning": {
                "type": "string",
                "description": "Optional concise strategic rationale.",
            },
        },
    }


def available_prompt_versions() -> tuple[str, ...]:
    return tuple(
        path.stem
        for path in sorted(PROMPTS_DIR.glob("*.md"))
        if not path.stem.startswith("character_select")
    )


@dataclass(frozen=True, slots=True)
class MatchHistoryEntry:
    """A single past match result for tournament character selection context."""

    your_character: str
    opponent_character: str
    result: str  # "win", "loss", or "draw"
    your_final_hp: int
    opponent_final_hp: int


def render_character_select_prompt(
    *,
    player_id: str,
    opponent_policy_id: str | None = None,
    match_history: list[MatchHistoryEntry] | None = None,
) -> str:
    """Render the character selection prompt, optionally with tournament history."""
    if not CHARACTER_SELECT_TEMPLATE.is_file():
        raise PromptTemplateError(
            f"character select template not found at {CHARACTER_SELECT_TEMPLATE}"
        )

    sections = [CHARACTER_SELECT_TEMPLATE.read_text(encoding="utf-8").strip()]

    if match_history:
        history_lines = ["\n## Tournament Match History"]
        if opponent_policy_id:
            history_lines.append(f"Your past results against **{opponent_policy_id}**:")
        for i, entry in enumerate(match_history, 1):
            history_lines.append(
                f"- Match {i}: You picked **{entry.your_character}**, "
                f"opponent picked **{entry.opponent_character}**. "
                f"{'You won' if entry.result == 'win' else 'You lost' if entry.result == 'loss' else 'Draw'} "
                f"({entry.your_final_hp} HP remaining vs {entry.opponent_final_hp} HP)."
            )
        sections.append("\n".join(history_lines))

    return "\n\n".join(sections).strip() + "\n"


def character_select_output_json_schema() -> JsonObject:
    """JSON schema for character selection structured output."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reasoning", "character"],
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Strategic reasoning for the character choice.",
            },
            "character": {
                "type": "string",
                "enum": list(VALID_CHARACTERS),
                "description": "The chosen character name.",
            },
        },
    }


def _variant_for_prompt_version(prompt_version: str) -> PromptTemplateVariant:
    if prompt_version.startswith("minimal_"):
        return PromptTemplateVariant.MINIMAL
    if prompt_version.startswith("strategic_"):
        return PromptTemplateVariant.STRATEGIC
    if prompt_version.startswith("few_shot_"):
        return PromptTemplateVariant.FEW_SHOT
    if prompt_version.startswith("reasoning_"):
        return PromptTemplateVariant.REASONING_ENABLED
    raise PromptTemplateError(f"prompt version {prompt_version!r} does not map to a known variant")


def _turn_context(request: DecisionRequest, *, policy_id: str | None) -> JsonObject:
    ctx: JsonObject = {
        "match_id": request.match_id,
        "turn_id": request.turn_id,
        "player_id": request.player_id,
        "deadline_ms": request.deadline_ms,
        "decision_type": request.decision_type.value,
    }
    # Only include non-null optional fields to reduce prompt size
    if request.trace_seed is not None:
        ctx["trace_seed"] = request.trace_seed
    if policy_id is not None:
        ctx["policy_id"] = policy_id
    return ctx


def _situation_summary(request: DecisionRequest) -> str:
    """Pre-compute a tactical situation summary so the LLM doesn't need to do arithmetic."""
    obs = request.observation
    fighters = obs.fighters
    if len(fighters) < 2:
        return ""

    me = next((f for f in fighters if f.id == request.player_id), fighters[0])
    opp = next((f for f in fighters if f.id != request.player_id), fighters[1])
    h_distance = abs(me.position.x - opp.position.x)
    if h_distance < 100:
        range_label = "POINT BLANK"
    elif h_distance < 200:
        range_label = "CLOSE"
    elif h_distance < 400:
        range_label = "MID"
    else:
        range_label = "FAR"

    # Detect repetition in recent history
    history = obs.history
    my_key = f"{request.player_id}_action"
    recent_actions: list[str] = []
    for entry in history[-5:]:
        d = entry.to_dict()
        action = d.get(my_key)
        if action:
            recent_actions.append(action)
    repetition_warning = ""
    if len(recent_actions) >= 3:
        last_3 = recent_actions[-3:]
        if len(set(last_3)) == 1:
            repetition_warning = (
                f"\n**WARNING: You have used {last_3[0]} for 3+ turns in a row. "
                f"Choose something DIFFERENT this turn.**"
            )

    lines = [
        "## Situation",
        f"- You are **{request.player_id}** ({me.character})",
        f"- Your HP: {me.hp}/{me.max_hp} | Opponent HP: {opp.hp}/{opp.max_hp}",
        f"- Horizontal distance: **{int(h_distance)} units ({range_label})**",
        f"- Your state: {me.current_state} | Opponent state: {opp.current_state}",
        f"- Meter: {me.meter} | Burst: {me.burst}",
    ]

    # Vertical positioning
    if abs(opp.position.y) > 30:
        opp_air = "high in the air" if abs(opp.position.y) > 150 else "airborne"
        lines.append(
            f"- Opponent is **{opp_air}** (y={int(opp.position.y)}). "
            f"Most ground attacks will miss an airborne opponent. "
            f"Use anti-airs (UpwardSwipe) or wait for them to land."
        )
    if abs(me.position.y) > 30:
        lines.append(
            f"- You are **airborne** (y={int(me.position.y)}). "
            f"Ground-only attacks are unavailable. Use aerial moves or wait to land."
        )

    # Range guidance with h_reach awareness
    range_guidance = _range_guidance(request, h_distance)
    if range_guidance:
        lines.append(range_guidance)

    if repetition_warning:
        lines.append(repetition_warning)
    return "\n".join(lines)


def _range_guidance(request: DecisionRequest, h_distance: float) -> str:
    """Generate a compact note about which attacks can/cannot reach the opponent."""
    character = _resolve_active_character(request)
    catalog = _load_move_catalog()
    universal = cast(JsonObject, catalog.get("_universal", {}))
    char_moves = cast(JsonObject, catalog.get(character, {})) if character else {}

    # Collect h_reach data for legal actions
    in_range: list[str] = []
    out_of_range: list[str] = []
    for action in request.legal_actions:
        entry = cast(
            JsonObject,
            char_moves.get(action.action, universal.get(action.action, {})),
        )
        if not isinstance(entry, dict):
            continue
        h_reach = entry.get("h_reach")
        if h_reach is None or not isinstance(h_reach, int | float):
            continue
        cat = entry.get("category", "")
        if cat in ("defense", "movement", "utility"):
            continue
        label = action.label or action.action
        if h_reach >= h_distance:
            in_range.append(f"{label}(~{int(h_reach)})")
        else:
            out_of_range.append(f"{label}(~{int(h_reach)})")

    if not in_range and not out_of_range:
        return ""

    # Find available movement options from the legal action set
    move_options = _available_movement_labels(request)
    move_hint = f" Use {', '.join(move_options)}." if move_options else ""

    parts: list[str] = []
    if out_of_range and h_distance > 100:
        parts.append(
            f"**OUT OF RANGE** at {int(h_distance)} units: {', '.join(out_of_range[:6])}. "
            f"These will whiff — close distance first.{move_hint}"
        )
    if in_range:
        parts.append(f"In range: {', '.join(in_range[:8])}")
    if not in_range and out_of_range:
        parts.append(
            f"**NO attacks in range.** You must use movement to close distance.{move_hint}"
        )
    return "- " + " | ".join(parts) if parts else ""


_MOVEMENT_ACTIONS = frozenset(
    {
        "DashForward",
        "DashBackward",
        "Dash",
        "ChargeDash",
        "Jump",
        "SuperJump",
        "DoubleJump",
        "Fall",
    }
)


def _available_movement_labels(request: DecisionRequest) -> list[str]:
    """Return the labels of movement actions currently available in the legal set."""
    labels = []
    for action in request.legal_actions:
        if action.action in _MOVEMENT_ACTIONS:
            labels.append(action.label or action.action)
    return labels


def _tactical_cheat_sheet(request: DecisionRequest) -> str:
    """Generate a short tactical suggestion based on opponent state and recent history."""
    obs = request.observation
    fighters = obs.fighters
    if len(fighters) < 2:
        return ""

    me = next((f for f in fighters if f.id == request.player_id), fighters[0])
    opp = next((f for f in fighters if f.id != request.player_id), fighters[1])
    h_distance = abs(me.position.x - opp.position.x)

    lines = ["## Tactical Cheat Sheet"]

    # Suggestion based on opponent's current state — now range-aware
    opp_state = opp.current_state.lower()
    is_close = h_distance < 100
    is_mid = 100 <= h_distance < 250
    if any(kw in opp_state for kw in ("attack", "slash", "kick", "punch", "combo", "shoot")):
        lines.append(
            "- Opponent is ATTACKING: **Block (ParryHigh), SpotDodge, or Roll to counter**"
        )
    elif any(kw in opp_state for kw in ("block", "parry", "guard")):
        if is_close:
            lines.append(
                "- Opponent is BLOCKING: **Grab to throw through their guard (you're close enough)**"
            )
        else:
            lines.append(
                "- Opponent is BLOCKING: Grab beats block, but you're too far for most grabs. "
                "**Close distance first, or use a long-range grab (Lasso, Vacuum) if available.**"
            )
    elif any(kw in opp_state for kw in ("grab", "throw", "lasso")):
        if is_close:
            lines.append(
                "- Opponent is GRABBING: **Any attack beats a grab — use a fast attack (you're in range)**"
            )
        elif is_mid:
            lines.append(
                "- Opponent is GRABBING: Attacks beat grabs, but **check your attack's h_reach** — "
                "you may be too far for melee. Use a ranged attack or close distance."
            )
        else:
            lines.append(
                "- Opponent is GRABBING at range: Their grab probably can't reach you either. "
                "**Use this opening to approach or zone with projectiles.**"
            )
    elif any(kw in opp_state for kw in ("whiff", "recovery", "landing", "endlag")):
        if is_close:
            lines.append(
                "- Opponent is in RECOVERY: **Punish with a high-damage move — they can't block**"
            )
        else:
            lines.append(
                "- Opponent is in RECOVERY: Good punish opportunity, but "
                "**only if you have an attack that reaches.** Close distance or use ranged attacks."
            )
    else:
        lines.append("- Opponent is NEUTRAL: **Mix unpredictably between attack, grab, and block**")

    # Suggestion based on opponent's recent history pattern
    opp_key = "p1_action" if request.player_id == "p2" else "p2_action"
    recent_opp_actions: list[str] = []
    for entry in obs.history[-5:]:
        d = entry.to_dict()
        action = d.get(opp_key)
        if action:
            recent_opp_actions.append(action)

    if len(recent_opp_actions) >= 3:
        last_3 = recent_opp_actions[-3:]
        if len(set(last_3)) == 1:
            lines.append(
                f"- Opponent used **{last_3[0]}** 3x in a row — they are predictable. "
                f"Choose the counter (block beats attacks, attack beats grabs, grab beats blocks)."
            )
        elif len(recent_opp_actions) >= 4:
            # Check for alternating pattern
            last_4 = recent_opp_actions[-4:]
            if last_4[0] == last_4[2] and last_4[1] == last_4[3] and last_4[0] != last_4[1]:
                lines.append(
                    f"- Opponent alternates **{last_4[0]}** / **{last_4[1]}** — "
                    f"break the pattern with a DODGE, ROLL, or unexpected option."
                )

    # Outcome-based suggestion from recent history
    my_key = f"{request.player_id}_outcome"
    recent_outcomes: list[str] = []
    for entry in obs.history[-3:]:
        d = entry.to_dict()
        outcome = d.get(my_key)
        if outcome:
            recent_outcomes.append(outcome)

    if recent_outcomes:
        hit_count = sum(1 for o in recent_outcomes if o == "hit" and o != "")
        blocked_count = sum(1 for o in recent_outcomes if o in ("blocked", "clashed"))
        neutral_count = sum(1 for o in recent_outcomes if o == "neutral")
        if hit_count >= 2:
            lines.append(
                "- Your attacks are LANDING — keep up the pressure but stay unpredictable."
            )
        elif blocked_count >= 2:
            lines.append(
                "- Your attacks are being BLOCKED — switch to grabs or try a different timing."
            )
        elif neutral_count >= 2 and h_distance > 100:
            move_opts = _available_movement_labels(request)
            move_hint = f" Use {', '.join(move_opts)}." if move_opts else ""
            lines.append(
                "- Your recent attacks are WHIFFING (neutral outcome at range). "
                f"**Stop attacking from this distance.** Close distance first, "
                f"THEN attack when in range.{move_hint}"
            )

    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


# Per-character strategic guidance that the generic prompt can't provide.
# Keyed by character name as it appears in the observation.
_CHARACTER_GUIDES: dict[str, str] = {
    "Wizard": (
        "## Character Guide: Wizard\n"
        "Wizard is a **zoner/spellcaster** — your goal is to control space with projectiles "
        "and spells, NOT to sit and block.\n"
        "- **At close range (<50 units)**: TomeSlap (h_reach 32, fastest), Kick, VileClutch/Grab. "
        "Do NOT just parry — parrying repeatedly is the #1 mistake.\n"
        "- **At mid range (50-120 units)**: BoltOfMagma (h_reach 75, high damage), IceSpikeGround "
        "(h_reach 55), ConjureWeapon (h_reach 85). These are Wizard's best tools.\n"
        "- **At long range (120+ units)**: MagicMissile (fullscreen projectile), FlameWave (h_reach 250 "
        "projectile), ConjureStorm (h_reach 104). Wizard EXCELS here — use ranged spells.\n"
        "- **Movement**: Walk Back to create zoning distance. If opponent closes in, "
        "use Gust to push them away. Liftoff covers ~200 units as a dash-attack.\n"
        "- **NEVER parry more than 2 turns in a row.** Wizard has the best ranged kit — USE IT.\n"
        "- **WARNING**: ManaStrike has only 12 unit h_reach and HIGH commitment (long animation). "
        "Avoid it unless point-blank. TomeSlap (32) is better at close range."
    ),
    "Cowboy": (
        "## Character Guide: Cowboy\n"
        "Cowboy is a **versatile all-rounder** with tools for every range.\n"
        "- **At close range (<50 units)**: Pommel (h_reach 28, fast, safe), Grab (50), "
        "VSlash (68, high damage), 3Combo (48). Mix attacks and grabs.\n"
        "- **At mid range (50-110 units)**: HSlash2 (h_reach 109), Stinger (82 + lunge), "
        "AnkleCutter (92). Cowboy's sword normals are excellent here.\n"
        "- **At long range (110+ units)**: LightningSliceNeutral (h_reach 147, travels through air), "
        "Brandish → Shoot (fullscreen gun), Lasso (250, long-range grab).\n"
        "- **Movement**: When >150 units, use Walk Forward or Dash to close into sword range. "
        "Super Dash covers big gaps with armor. Don't spam LightningSlice from fullscreen — "
        "close distance and use higher-damage melee instead.\n"
        "- **Defense**: SpotDodge (invuln frames 5-7, fast recovery) and Foresight are strong.\n"
        "- Cowboy wins by mixing all three ranges. Don't camp one distance."
    ),
    "Robot": (
        "## Character Guide: Robot\n"
        "Robot is a **grappler/heavyweight** — your goal is to get close and land devastating "
        "command grabs and heavy attacks.\n"
        "- **Closing distance is CRITICAL.** Most Robot attacks have short range. Use Walk Forward, "
        "Dash, and Super Dash (armored — goes through single-hit attacks) to get in.\n"
        "- **At close range**: Vacuum (h_reach 140 — Robot's best grab, great range for a grab), "
        "Slap (85, fast poke), Blast (54, safe). Mix grabs with attacks.\n"
        "- **At range**: EyeBeam (h_reach 80-200, long range), Missile/LOIC (fullscreen supers). "
        "Use these to force the opponent to approach you.\n"
        "- Robot's attacks are slow but hit HARD. Jump to approach over projectiles."
    ),
    "Ninja": (
        "## Character Guide: Ninja\n"
        "Ninja is a **rushdown/mixup** character — stay close and overwhelm with fast attacks.\n"
        "- **Getting in**: Dash (projectile invuln frames 0-4, hops lows), SlideKick (travels ~150 "
        "units along ground), DropKick (lunges ~150 units). Use Shuriken at range to force reactions.\n"
        "- **At close range**: NunChukLight (h_reach 51, fast), GroundedPunch (33, fastest), "
        "Uppercut (28, huge damage launcher), Grab. Mix attacks and grabs.\n"
        "- **Key combos**: GroundedPunch → NunChukLight → Uppercut for big damage.\n"
        "- **Movement**: Ninja's speed advantage only works up close. If far, use Dash or "
        "Walk Forward to close in. GrapplingHook repositions from range. Don't throw short-range "
        "attacks from mid range — they WILL whiff."
    ),
    "Mutant": (
        "## Character Guide: Mutant\n"
        "Mutant is an **aggressive rushdown** character with acid damage-over-time.\n"
        "- **Getting in**: AcidSlashJ (Leap Slash, travels ~150 units, 10-hit), "
        "DashThroughAttack (Sneak Attack, dashes through opponent), WallTrick (Pounce, ~200 units). "
        "Also use Dash/Walk Forward — Mutant NEEDS to be close.\n"
        "- **At close range (<60 units)**: Swipe (h_reach 44, fast), Sweep (49, knockdown), "
        "BiteGrab (40, applies poison DoT — grabs beat blocking). Mix Swipe/Grab unpredictably.\n"
        "- **At range**: CausticSpike (250, acid projectile). Use to force opponent to approach.\n"
        "- **Key strategy**: Apply acid with AcidSlashH/BiteGrab, then stay aggressive. Acid DoT "
        "does damage over time so pressure counts. GroundToAirSpin (Shred, 9 hits) for big damage.\n"
        "- **Movement is essential.** Most Mutant attacks are short range. If >60 units away, "
        "close distance before attacking. Don't throw Swipe at 150 units — it will whiff.\n"
        "- Mutant is NOT a defensive character. Blocking wastes your speed advantage."
    ),
}


def _character_guide(request: DecisionRequest) -> str:
    """Return character-specific strategic guidance if available."""
    character = _resolve_active_character(request)
    if character and character in _CHARACTER_GUIDES:
        return _CHARACTER_GUIDES[character]
    return ""


def _compact_observation(request: DecisionRequest) -> JsonObject:
    """Produce a compact observation dict that strips redundant/verbose fields."""
    obs = request.observation.to_dict()

    # Add pre-computed distances so the model doesn't have to do coordinate math
    fighters = obs.get("fighters", [])
    if len(fighters) >= 2:
        f0 = fighters[0]
        f1 = fighters[1]
        p0 = f0.get("position", {})
        p1 = f1.get("position", {})
        h_dist = abs(p0.get("x", 0) - p1.get("x", 0))
        v_diff = p0.get("y", 0) - p1.get("y", 0)  # positive = f0 above f1
        obs["_distances"] = {
            "horizontal": int(h_dist),
            "vertical_diff": int(v_diff),
            "note": "horizontal = abs(x difference). "
            "vertical_diff = your_y - opponent_y (negative means opponent is higher). "
            "Compare horizontal to h_reach to know if an attack can connect.",
        }

    # Strip history from observation — it's already summarized in Situation/Cheat Sheet
    # and the full entries are verbose. Keep last 5 entries with compact format.
    if "history" in obs and isinstance(obs["history"], list):
        compact_history = []
        for entry in obs["history"][-5:]:
            if not isinstance(entry, dict):
                continue
            compact: JsonObject = {}
            if "game_tick" in entry:
                compact["tick"] = entry["game_tick"]
            elif "turn_id" in entry:
                compact["turn_id"] = entry["turn_id"]
            for key in (
                "p1_action",
                "p2_action",
                "p1_hp",
                "p2_hp",
                "p1_hp_delta",
                "p2_hp_delta",
                "p1_outcome",
                "p2_outcome",
            ):
                if key in entry and entry[key] is not None:
                    compact[key] = entry[key]
            compact_history.append(compact)
        obs["history"] = compact_history
    return obs


_DEFENSIVE_ACTIONS = frozenset(
    {
        "ParryHigh",
        "ParryLow",
        "ParrySuper",
        "ParryAfterWhiff",
        "Block",
        "BlockHigh",
        "BlockLow",
    }
)

_REPETITION_FILTER_THRESHOLD = 4


def _detect_repeated_action(request: DecisionRequest) -> str | None:
    """Return action name if the player has repeated a defensive action N+ times in a row."""
    history = request.observation.history
    my_key = f"{request.player_id}_action"
    recent: list[str] = []
    for entry in history[-_REPETITION_FILTER_THRESHOLD:]:
        d = entry.to_dict()
        action = d.get(my_key)
        if action:
            recent.append(action)

    if len(recent) >= _REPETITION_FILTER_THRESHOLD and len(set(recent)) == 1:
        action = recent[0]
        if action in _DEFENSIVE_ACTIONS:
            return action
    return None


def _grouped_legal_actions_section(request: DecisionRequest) -> str:
    """Render legal actions grouped by category instead of a flat list."""
    actions = _legal_actions_payload(request)

    # Mechanically remove defensive actions repeated 4+ times in a row.
    # The model ignores prompt warnings about repetition, so we remove the
    # option entirely to force it to pick something else.
    banned_action = _detect_repeated_action(request)
    if banned_action:
        actions = [a for a in actions if a.get("action") != banned_action]

    groups: dict[str, list[JsonObject]] = {}
    for action in actions:
        cat = str(action.get("category", "other"))
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(action)

    # Preferred category order
    order = ["offense", "defense", "grab", "movement", "special", "super", "utility", "other"]
    lines = ["## Legal Actions"]
    for cat in order:
        if cat not in groups:
            continue
        lines.append(f"\n### {cat.title()}")
        lines.append(f"```json\n{_json_dump(groups[cat])}\n```")
        del groups[cat]
    # Any remaining categories
    for cat, acts in groups.items():
        lines.append(f"\n### {cat.title()}")
        lines.append(f"```json\n{_json_dump(acts)}\n```")
    return "\n".join(lines)


def _legal_actions_payload(request: DecisionRequest) -> list[JsonObject]:
    # Resolve character name from the active player's fighter observation
    character_name = _resolve_active_character(request)
    catalog = _load_move_catalog()
    universal_moves = cast(JsonObject, catalog.get("_universal", {}))
    character_moves = cast(JsonObject, catalog.get(character_name, {})) if character_name else {}

    result: list[JsonObject] = []
    for action in request.legal_actions:
        # Look up catalog entry: character-specific first, then universal
        catalog_entry = cast(
            JsonObject,
            character_moves.get(action.action, universal_moves.get(action.action, {})),
        )

        entry: JsonObject = {
            "action": action.action,
            "label": action.label,
        }

        # Inject catalog description and tactical metadata
        desc = action.description
        if not desc and isinstance(catalog_entry, dict):
            desc = catalog_entry.get("description")
        if desc:
            entry["description"] = desc

        cat = catalog_entry.get("category") if isinstance(catalog_entry, dict) else None
        if cat:
            entry["category"] = cat
        spd = catalog_entry.get("speed") if isinstance(catalog_entry, dict) else None
        if spd:
            entry["speed"] = spd
        dmg = catalog_entry.get("damage") if isinstance(catalog_entry, dict) else None
        if dmg:
            entry["damage"] = dmg
        rng = catalog_entry.get("range") if isinstance(catalog_entry, dict) else None
        if rng:
            entry["range"] = rng
        beats = catalog_entry.get("beats") if isinstance(catalog_entry, dict) else None
        if beats:
            entry["beats"] = beats
        weakness = catalog_entry.get("weakness") if isinstance(catalog_entry, dict) else None
        if weakness:
            entry["weakness"] = weakness
        h_reach = catalog_entry.get("h_reach") if isinstance(catalog_entry, dict) else None
        if h_reach is not None:
            entry["h_reach"] = h_reach
        v_range = catalog_entry.get("v_range") if isinstance(catalog_entry, dict) else None
        if v_range is not None:
            entry["v_range"] = v_range
        commitment = catalog_entry.get("commitment") if isinstance(catalog_entry, dict) else None
        if commitment:
            entry["commitment"] = commitment
        startup_ticks = (
            catalog_entry.get("startup_ticks") if isinstance(catalog_entry, dict) else None
        )
        if startup_ticks is not None:
            entry["startup_ticks"] = startup_ticks
        active_ticks_val = (
            catalog_entry.get("active_ticks") if isinstance(catalog_entry, dict) else None
        )
        if active_ticks_val is not None:
            entry["active_ticks"] = active_ticks_val
        total_damage = (
            catalog_entry.get("total_damage") if isinstance(catalog_entry, dict) else None
        )
        if total_damage is not None:
            entry["total_damage"] = total_damage

        # Only include non-trivial payload_specs (omit zero-range XY plots)
        if action.payload_spec and not _is_zero_range_payload(action.payload_spec):
            entry["payload_spec"] = action.payload_spec
        if action.prediction_spec:
            entry["prediction_spec"] = action.prediction_spec
        # Only show supports flags that are true to save tokens
        true_supports = {k: v for k, v in action.supports.to_dict().items() if v}
        if true_supports:
            entry["supports"] = true_supports
        result.append(entry)
    return result


def _is_zero_range_payload(payload_spec: JsonObject) -> bool:
    """Check if all fields in a payload_spec have zero-choice ranges and can be auto-defaulted.

    Returns True only if every field is either:
    - A numeric field with min == max (no real choice)
    - An XY field where both axes have min == max
    - A boolean field (defaults to false)

    When True, the prompt omits the payload_spec and the response parser auto-fills defaults.
    """
    props = payload_spec.get("properties")
    if not isinstance(props, dict) or not props:
        return False
    for field in props.values():
        if not isinstance(field, dict):
            continue
        field_type = field.get("type")
        # Boolean fields always default to false — zero-choice
        if field_type == "boolean":
            continue
        field_min = field.get("minimum")
        field_max = field.get("maximum")
        if field_min is not None and field_max is not None and field_min == field_max:
            continue  # Zero range numeric
        # Check nested xy properties
        nested_props = field.get("properties")
        if isinstance(nested_props, dict):
            all_zero = True
            for nested in nested_props.values():
                if not isinstance(nested, dict):
                    continue
                n_min = nested.get("minimum")
                n_max = nested.get("maximum")
                if n_min is None or n_max is None or n_min != n_max:
                    all_zero = False
                    break
            if all_zero:
                continue
        # Has a real choice — not zero-range
        return False
    return True


def _resolve_active_character(request: DecisionRequest) -> str | None:
    """Resolve the character name for the active player from the observation."""
    for fighter in request.observation.fighters:
        if fighter.id == request.player_id:
            # Character name from observation (e.g. "P1", "Ninja", "Mutant")
            name = fighter.character
            # The game uses generic names like "P1"/"P2" when character_selection
            # is mirror mode; try to match against known catalog characters
            if name and name not in ("P1", "P2"):
                return name
    return None


def _load_move_catalog() -> JsonObject:
    """Load and cache the move catalog from prompts/move_catalog.json."""
    global _move_catalog_cache  # noqa: PLW0603
    if _move_catalog_cache is not None:
        return _move_catalog_cache
    if not MOVE_CATALOG_PATH.is_file():
        _move_catalog_cache = {}
        return _move_catalog_cache
    raw = json.loads(MOVE_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _move_catalog_cache = {}
        return _move_catalog_cache
    _move_catalog_cache = cast(JsonObject, raw)
    return _move_catalog_cache


def _json_dump(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
