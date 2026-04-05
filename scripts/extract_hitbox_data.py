"""Extract hitbox data from decompiled .tscn files and merge into move_catalog.json.

Parses Godot .tscn scene files for each character, extracts Hitbox node properties
(geometry, hitstun, damage, frame data), and maps them back to their parent move state.

Also parses projectile .tscn files to capture hitbox data for moves whose damage
comes from spawned projectiles rather than inline hitboxes.

Extracts state-level data (anim_length, iasa_at, selectable) and filters
non-selectable internal states from the output.

The AABB for a CollisionBox is:
    x1 = (x - width) + pos_x
    x2 = (x + width) + pos_x
    y1 = (y - height) + pos_y
    y2 = (y + height) + pos_y

where x, y are the center offset and width, height are half-extents.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DECOMPILE_ROOT = (
    Path(__file__).resolve().parent.parent / "docs" / "decompile-output" / "project"
)

# Character name -> tscn file path (relative to DECOMPILE_ROOT)
CHARACTER_TSCN_FILES: dict[str, str] = {
    "Wizard": "characters/wizard/Wizard.tscn",
    "Cowboy": "characters/swordandgun/SwordGuy.tscn",
    "Robot": "characters/robo/Robot.tscn",
    "Ninja": "characters/stickman/NinjaGuy.tscn",
    "Mutant": "characters/mutant/Mutant.tscn",
}

BASE_CHAR_TSCN = "characters/BaseChar.tscn"

# ---------------------------------------------------------------------------
# Projectile mapping: (character, move_state) -> list of projectile tscn paths
# relative to DECOMPILE_ROOT. When a move spawns a projectile whose hitbox is
# not in the character's main .tscn, we parse these files instead.
# ---------------------------------------------------------------------------
PROJECTILE_MOVE_MAP: dict[tuple[str, str], list[str]] = {
    # ── Cowboy ──
    ("Cowboy", "1000Cuts"): [
        "characters/swordandgun/projectiles/1000cuts/1000Cuts.tscn",
    ],
    ("Cowboy", "1000CutsEnd"): [
        "characters/swordandgun/projectiles/1000cuts/1000Cuts.tscn",
    ],
    ("Cowboy", "GunThrow"): [
        "characters/swordandgun/projectiles/Gun.tscn",
    ],
    ("Cowboy", "Shoot2"): [
        "characters/swordandgun/projectiles/NewBullet.tscn",
    ],
    ("Cowboy", "ShootDodge2"): [
        "characters/swordandgun/projectiles/NewBullet.tscn",
    ],
    ("Cowboy", "PointBlank"): [
        "characters/swordandgun/projectiles/NewBullet.tscn",
    ],
    ("Cowboy", "TimeBullet"): [
        "characters/swordandgun/projectiles/frozen_bullet.tscn",
        "characters/swordandgun/projectiles/NewTimeBullet.tscn",
    ],
    ("Cowboy", "TimeSplitter"): [
        "characters/swordandgun/projectiles/frozen_bullet.tscn",
        "characters/swordandgun/projectiles/NewTimeBullet.tscn",
    ],
    ("Cowboy", "Lasso"): [
        "characters/swordandgun/projectiles/Lasso.tscn",
    ],
    ("Cowboy", "Foresight"): [
        "characters/swordandgun/projectiles/AfterImageExplosion.tscn",
    ],
    ("Cowboy", "ForesightNeutral"): [
        "characters/swordandgun/projectiles/AfterImageExplosion.tscn",
    ],
    # ── Robot ──
    ("Robot", "Grenade"): [
        "characters/robo/projectiles/RobotGrenade.tscn",
        "characters/robo/projectiles/NadeExplosion.tscn",
    ],
    ("Robot", "Missile"): [
        "characters/robo/projectiles/Missile.tscn",
    ],
    ("Robot", "LOIC"): [
        "characters/robo/projectiles/orbital_strike.tscn",
    ],
    ("Robot", "Flamethrower"): [
        "characters/robo/projectiles/GroundFire.tscn",
    ],
    ("Robot", "Flamethrower2"): [
        "characters/robo/projectiles/GroundFireLoic.tscn",
    ],
    # ── Ninja ──
    ("Ninja", "Fireball"): [
        "characters/stickman/projectiles/StickmanFireball.tscn",
    ],
    ("Ninja", "AirFireball"): [
        "characters/stickman/projectiles/StickmanAirFireball.tscn",
    ],
    ("Ninja", "GrapplingHook"): [
        "characters/stickman/projectiles/grappling_hook.tscn",
    ],
    ("Ninja", "StickyBomb"): [
        "characters/stickman/projectiles/StickyBomb.tscn",
    ],
    ("Ninja", "StickyBombThrow"): [
        "characters/stickman/projectiles/StickyBomb.tscn",
    ],
    ("Ninja", "Caltrops"): [
        "characters/stickman/projectiles/Caltrops.tscn",
    ],
    ("Ninja", "Summon"): [
        "characters/stickman/projectiles/Stackriken.tscn",
    ],
    ("Ninja", "WhipWave"): [
        "characters/stickman/projectiles/WhipWave.tscn",
    ],
    # ── Wizard ──
    ("Wizard", "FlameWave"): [
        "characters/wizard/projectiles/FlameWave.tscn",
    ],
    ("Wizard", "MagicMissile"): [
        "characters/wizard/projectiles/MagicMissile.tscn",
    ],
    ("Wizard", "SparkBomb"): [
        "characters/wizard/projectiles/SparkBomb.tscn",
    ],
    ("Wizard", "VileClutch"): [
        "characters/wizard/projectiles/VileClutch.tscn",
    ],
    ("Wizard", "Geyser"): [
        "characters/wizard/projectiles/GeyserProjectile.tscn",
    ],
    ("Wizard", "Orb"): [
        "characters/wizard/projectiles/orb/Orb.tscn",
    ],
    ("Wizard", "OrbPush"): [
        "characters/wizard/projectiles/OrbDart.tscn",
    ],
    ("Wizard", "Launch"): [
        "characters/wizard/projectiles/telekinesis/TelekinesisBoulder.tscn",
    ],
    ("Wizard", "SuperLaunch"): [
        "characters/wizard/projectiles/telekinesis/TelekinesisBoulder.tscn",
    ],
    ("Wizard", "Telekinesis"): [
        "characters/wizard/projectiles/telekinesis/TelekinesisBoulder.tscn",
    ],
    # ── Mutant ──
    ("Mutant", "CausticSpike"): [
        "characters/mutant/projectiles/CausticSpike.tscn",
    ],
    ("Mutant", "CausticBarrage"): [
        "characters/mutant/projectiles/CausticThorn.tscn",
    ],
    ("Mutant", "CausticBarrageAir"): [
        "characters/mutant/projectiles/CausticThorn.tscn",
    ],
    ("Mutant", "Shockwave"): [
        "characters/mutant/projectiles/Shockwave.tscn",
    ],
}

# Default values from Hitbox.gd and CollisionBox.gd
HITBOX_DEFAULTS: dict[str, int | str] = {
    "x": 0,
    "y": 0,
    "width": 20,
    "height": 20,
    "damage": 0,
    "hitstun_ticks": 30,
    "combo_hitstun_ticks": -1,  # means "same as hitstun_ticks"
    "start_tick": 0,
    "active_ticks": 5,
    "hit_height": 1,  # 0=High, 1=Mid, 2=Low
    "knockback": "10.0",
    "dir_x": "1.0",
    "dir_y": "-1.0",
    "knockdown": False,
    "hard_knockdown": False,
    "hits_vs_aerial": True,
    "hits_vs_grounded": True,
    "hits_vs_standing": True,
    "hits_otg": False,
    "plus_frames": 0,
    "damage_in_combo": -1,
    "guard_break": False,
}

# Properties we want to extract from hitbox nodes
EXTRACT_PROPS = {
    "x",
    "y",
    "width",
    "height",
    "damage",
    "damage_in_combo",
    "minimum_damage",
    "hitstun_ticks",
    "combo_hitstun_ticks",
    "start_tick",
    "active_ticks",
    "hit_height",
    "knockback",
    "dir_x",
    "dir_y",
    "knockdown",
    "hard_knockdown",
    "hits_vs_aerial",
    "hits_vs_grounded",
    "hits_vs_standing",
    "hits_otg",
    "plus_frames",
    "guard_break",
    "hitbox_type",
}

# State-level properties to extract from move state nodes
STATE_PROPS = {
    "anim_length",
    "iasa_at",
    "type",
    "title",
    "selectable",
}

HIT_HEIGHT_MAP = {0: "high", 1: "mid", 2: "low"}

# Moves where the .gd state script programmatically repositions hitboxes at runtime.
# These override the static .tscn values. Format:
#   (character, move_state) -> override dict to merge into the summary
PROGRAMMATIC_OVERRIDES: dict[tuple[str, str], dict] = {
    # ManaStrike.gd: hitbox x is set dynamically based on slider input
    # x = round(lerp(MIN_DISTANCE=30, MAX_DISTANCE=120, slider)) * facing
    # hitbox width=12, so effective h_range is (30-12)=18 to (120+12)=132
    ("Wizard", "ManaStrike"): {
        "h_range_min": 18,
        "h_range_max": 132,
    },
    # TimeSplitter: 12 hitboxes at (0,0) in .tscn — positions set dynamically
    # by .gd at runtime spread across the stage. Effective fullscreen.
    ("Cowboy", "TimeSplitter"): {
        "h_range_min": 0,
        "h_range_max": 550,
    },
    # Shoot2/PointBlank: bullet hitbox is a travelling projectile.
    # Point-blank has barrel offset ~25-40 units, bullet travels fullscreen.
    ("Cowboy", "Shoot2"): {
        "h_range_min": 0,
        "h_range_max": 550,
    },
    ("Cowboy", "ShootDodge2"): {
        "h_range_min": 0,
        "h_range_max": 550,
    },
    ("Cowboy", "PointBlank"): {
        "h_range_min": 0,
        "h_range_max": 80,
    },
    # TimeBullet: frozen bullet tracks to opponent, then fires. Effective fullscreen.
    ("Cowboy", "TimeBullet"): {
        "h_range_min": 0,
        "h_range_max": 550,
    },
    # Lasso: projectile travels forward with gravity. ~200 unit effective range.
    ("Cowboy", "Lasso"): {
        "h_range_min": 0,
        "h_range_max": 200,
    },
    # GunThrow: thrown gun arcs forward. ~250 unit effective range.
    ("Cowboy", "GunThrow"): {
        "h_range_min": 0,
        "h_range_max": 250,
    },
    # Foresight/ForesightNeutral: places marker at current position, explodes later.
    # Effective range is where the marker was placed (25 unit explosion radius).
    ("Cowboy", "Foresight"): {
        "h_range_min": 0,
        "h_range_max": 25,
    },
    ("Cowboy", "ForesightNeutral"): {
        "h_range_min": 0,
        "h_range_max": 25,
    },
    # Robot Missile: homing projectile, effective fullscreen.
    ("Robot", "Missile"): {
        "h_range_min": 0,
        "h_range_max": 550,
    },
    # Robot Grenade: thrown arc + explosion radius ~42. ~300 effective range.
    ("Robot", "Grenade"): {
        "h_range_min": 0,
        "h_range_max": 300,
    },
    # Robot LOIC: full-height column laser. Fullscreen.
    ("Robot", "LOIC"): {
        "h_range_min": 0,
        "h_range_max": 550,
    },
    # Robot Flamethrower: ground fire, ~100 unit width from character.
    ("Robot", "Flamethrower"): {
        "h_range_min": 0,
        "h_range_max": 100,
    },
    ("Robot", "Flamethrower2"): {
        "h_range_min": 0,
        "h_range_max": 100,
    },
    # Ninja Fireball: travels fullscreen.
    ("Ninja", "Fireball"): {
        "h_range_min": 0,
        "h_range_max": 550,
    },
    ("Ninja", "AirFireball"): {
        "h_range_min": 0,
        "h_range_max": 550,
    },
    # Ninja GrapplingHook: travels with gravity. ~300 range.
    ("Ninja", "GrapplingHook"): {
        "h_range_min": 0,
        "h_range_max": 300,
    },
    # Ninja StickyBomb: thrown arc. ~250 range.
    ("Ninja", "StickyBomb"): {
        "h_range_min": 0,
        "h_range_max": 250,
    },
    ("Ninja", "StickyBombThrow"): {
        "h_range_min": 0,
        "h_range_max": 250,
    },
    # Ninja Caltrops: trap placed near character. ~50 range.
    ("Ninja", "Caltrops"): {
        "h_range_min": 0,
        "h_range_max": 50,
    },
    # Ninja Summon: orb that orbits and fires kunai. ~110 range (orb radius).
    ("Ninja", "Summon"): {
        "h_range_min": 0,
        "h_range_max": 110,
    },
    # Wizard MagicMissile: homing dart, fullscreen.
    ("Wizard", "MagicMissile"): {
        "h_range_min": 0,
        "h_range_max": 550,
    },
    # Wizard SparkBomb: placed trap. ~26 explosion radius.
    ("Wizard", "SparkBomb"): {
        "h_range_min": 0,
        "h_range_max": 100,
    },
    # Wizard FlameWave: travels forward. ~200 range.
    ("Wizard", "FlameWave"): {
        "h_range_min": 0,
        "h_range_max": 200,
    },
    # Wizard VileClutch: rising claw at set distance.
    ("Wizard", "VileClutch"): {
        "h_range_min": 0,
        "h_range_max": 32,
    },
    # Wizard Geyser: ground eruption at medium range.
    ("Wizard", "Geyser"): {
        "h_range_min": 0,
        "h_range_max": 150,
    },
    # Wizard Telekinesis/Launch/SuperLaunch: boulder, travels forward. ~300 range.
    ("Wizard", "Telekinesis"): {
        "h_range_min": 0,
        "h_range_max": 300,
    },
    ("Wizard", "Launch"): {
        "h_range_min": 0,
        "h_range_max": 300,
    },
    ("Wizard", "SuperLaunch"): {
        "h_range_min": 0,
        "h_range_max": 300,
    },
    # Mutant CausticSpike: ground eruption spread.
    ("Mutant", "CausticSpike"): {
        "h_range_min": 0,
        "h_range_max": 127,
    },
    # Mutant CausticBarrage: thorn projectile.
    ("Mutant", "CausticBarrage"): {
        "h_range_min": 0,
        "h_range_max": 550,
    },
    ("Mutant", "CausticBarrageAir"): {
        "h_range_min": 0,
        "h_range_max": 550,
    },
    # Mutant Shockwave: ground wave, travels forward. ~300 range.
    ("Mutant", "Shockwave"): {
        "h_range_min": 0,
        "h_range_max": 300,
    },
}

# Moves that have no hitbox by design (stances, utility, movement, etc.)
# but are in the catalog. We mark them explicitly so the prompt system knows
# not to make range predictions for them.
NO_HITBOX_MOVES: dict[tuple[str, str], str] = {
    ("Cowboy", "Brandish"): "Quick Draw stance entry",
    ("Cowboy", "QuickerDraw"): "Quick Draw follow-up stance",
    ("Cowboy", "Holster"): "Exit Quick Draw stance",
    ("Cowboy", "SlowHolster"): "Slow holster animation",
    ("Cowboy", "GunReel"): "Pull thrown gun back (no damage)",
    ("Cowboy", "FastTeleport"): "Teleport (invincible movement)",
    ("Cowboy", "SpotDodge"): "Spot dodge (invincible)",
    ("Cowboy", "LassoReel"): "Izuna Drop follow-up (grab, no hitbox)",
    ("Robot", "DisableLoic"): "LOIC shutdown (no damage)",
    ("Robot", "Magnetize"): "Magnetize pull (no damage hitbox)",
    ("Robot", "Drive"): "Enter drive vehicle stance",
    ("Ninja", "StoreMomentum"): "Store momentum resource",
    ("Ninja", "BackSway"): "Backwards dodge",
    ("Ninja", "Substitution"): "Counter/dodge",
    ("Wizard", "OrbTeleport"): "Teleport to orb position",
    ("Wizard", "OrbTether"): "Tether to orb (movement)",
    ("Wizard", "Gust"): "Wind push (displacement, no damage)",
    ("Mutant", "JukeCharge"): "Howl buff (no damage)",
    ("Mutant", "WallTrick"): "Pounce to wall (no initial hitbox)",
    ("Mutant", "BiteGrab"): "Envenom grab (grab, no hitbox)",
}


@dataclass
class HitboxData:
    """Raw hitbox data extracted from a .tscn node."""

    node_name: str
    move_state: str
    props: dict[str, int | float | str | bool] = field(default_factory=dict)


@dataclass
class StateData:
    """State-level data extracted from a move state node."""

    state_name: str
    anim_length: int | None = None
    iasa_at: int | None = None
    move_type: int | None = None  # 1=normal, 2=special, 3=super, 4=dodge
    title: str | None = None
    selectable: bool = True  # defaults to true if not specified


@dataclass
class MoveHitboxSummary:
    """Summarized hitbox coverage for a single move."""

    move_state: str
    # Horizontal range from character center (facing right)
    h_range_min: int  # closest horizontal distance that hits
    h_range_max: int  # farthest horizontal distance that hits
    # Vertical coverage relative to character position
    v_top: int  # highest point hit (most negative y = highest)
    v_bottom: int  # lowest point hit
    # Hit height category
    hit_height: str  # "high", "mid", "low", or "all"
    # Which targets it can hit
    hits_aerial: bool
    hits_grounded: bool
    # Frame data
    startup_ticks: int
    active_ticks: int
    total_active_ticks: int  # sum across all hitboxes
    # Damage
    damage: int
    total_damage: int  # sum if multi-hit with different groups
    # Hitstun
    hitstun_ticks: int
    combo_hitstun_ticks: int
    # Knockback
    knockback: float
    knockdown: bool
    # Frame advantage on block
    plus_frames: int
    # State-level data
    anim_length: int | None = None
    iasa_at: int | None = None
    move_type: int | None = None
    title: str | None = None
    selectable: bool = True
    # Source of hitbox data
    source: str = "character"  # "character", "projectile", "override", "no_hitbox"
    source_note: str | None = None
    # Raw hitbox list for detailed inspection
    hitboxes: list[dict] = field(default_factory=list)


def parse_tscn_nodes(tscn_path: Path) -> list[dict]:
    """Parse a .tscn file and return a list of node dicts with their properties."""
    text = tscn_path.read_text()
    nodes: list[dict] = []
    current_node: dict | None = None

    # Match node headers like: [node name="Hitbox" type="Node2D" parent="StateMachine/ManaStrike" index="0"]
    node_re = re.compile(r'\[node name="([^"]+)".*?parent="([^"]*)"')
    # Match property lines like: damage = 80  or  dir_x = "1.0"  or  knockdown = true
    prop_re = re.compile(r"^(\w+)\s*=\s*(.+)$")

    for line in text.splitlines():
        line = line.strip()

        # New section header (node, ext_resource, sub_resource, etc.)
        if line.startswith("["):
            # Save previous node
            if current_node is not None:
                nodes.append(current_node)
                current_node = None

            m = node_re.match(line)
            if m:
                current_node = {
                    "_name": m.group(1),
                    "_parent": m.group(2),
                }
            continue

        # Property line within a node
        if current_node is not None and line:
            m = prop_re.match(line)
            if m:
                key, raw_val = m.group(1), m.group(2)
                current_node[key] = _parse_value(raw_val)

    # Don't forget the last node
    if current_node is not None:
        nodes.append(current_node)

    return nodes


def _parse_value(raw: str) -> int | float | str | bool:
    """Parse a .tscn property value."""
    raw = raw.strip()
    if raw == "true":
        return True
    if raw == "false":
        return False
    # Quoted string
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    # Integer
    try:
        return int(raw)
    except ValueError:
        pass
    # Float
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def extract_state_data_from_tscn(tscn_path: Path) -> dict[str, StateData]:
    """Extract state-level data (anim_length, iasa_at, etc.) from move state nodes."""
    nodes = parse_tscn_nodes(tscn_path)
    states: dict[str, StateData] = {}

    for node in nodes:
        parent = node.get("_parent", "")
        name = node.get("_name", "")

        # State nodes are direct children of StateMachine
        if parent != "StateMachine":
            continue

        sd = StateData(state_name=name)
        if "anim_length" in node:
            val = node["anim_length"]
            if isinstance(val, int):
                sd.anim_length = val
        if "iasa_at" in node:
            val = node["iasa_at"]
            if isinstance(val, int):
                sd.iasa_at = val
        if "type" in node:
            val = node["type"]
            if isinstance(val, int):
                sd.move_type = val
        if "title" in node:
            val = node["title"]
            if isinstance(val, str):
                sd.title = val
        if "selectable" in node:
            sd.selectable = bool(node["selectable"])
        # default is true if not specified

        states[name] = sd

    return states


def extract_hitboxes_from_tscn(tscn_path: Path) -> dict[str, list[HitboxData]]:
    """Extract hitbox data grouped by move state from a .tscn file.

    Returns a dict mapping move_state_name -> [HitboxData, ...].
    Only includes nodes that are children of StateMachine/<MoveName>/...
    and that have hitbox-relevant properties (damage, width, etc.).
    """
    nodes = parse_tscn_nodes(tscn_path)
    move_hitboxes: dict[str, list[HitboxData]] = {}

    for node in nodes:
        parent = node.get("_parent", "")
        name = node.get("_name", "")

        # We want hitbox nodes under StateMachine/<MoveName>/...
        # The parent path tells us which move this hitbox belongs to.
        if not parent.startswith("StateMachine/"):
            continue

        # Extract the move state name (first path component after StateMachine/)
        parts = parent.split("/")
        if len(parts) < 2:
            continue
        move_state = parts[1]

        # Filter to actual hitbox nodes (must have damage or be named *Hitbox*)
        is_hitbox = (
            "Hitbox" in name
            or "damage" in node
            or "hitstun_ticks" in node
            or "start_tick" in node
        )
        # Exclude non-hitbox collision boxes (ThrowBox, WindBox, HurtboxState)
        is_excluded = any(
            excl in name
            for excl in ("ThrowBox", "WindBox", "HurtboxState", "CollisionBox")
        )

        if not is_hitbox or is_excluded:
            continue

        # Extract relevant properties
        props: dict[str, int | float | str | bool] = {}
        for key in EXTRACT_PROPS:
            if key in node:
                props[key] = node[key]
            elif key in HITBOX_DEFAULTS:
                props[key] = HITBOX_DEFAULTS[key]

        hb = HitboxData(node_name=name, move_state=move_state, props=props)
        move_hitboxes.setdefault(move_state, []).append(hb)

    return move_hitboxes


def extract_hitboxes_from_projectile(tscn_path: Path) -> list[HitboxData]:
    """Extract hitbox data from a projectile .tscn file.

    Projectile files have a different structure: hitboxes can be at root level
    or under state nodes (e.g., StateMachine/Default/Hitbox, or just Hitbox).
    We collect all hitbox nodes regardless of parent path.
    """
    if not tscn_path.exists():
        return []

    nodes = parse_tscn_nodes(tscn_path)
    hitboxes: list[HitboxData] = []

    for node in nodes:
        name = node.get("_name", "")

        is_hitbox = (
            "Hitbox" in name
            or "damage" in node
            or "hitstun_ticks" in node
        )
        is_excluded = any(
            excl in name
            for excl in ("ThrowBox", "WindBox", "HurtboxState", "CollisionBox")
        )

        if not is_hitbox or is_excluded:
            continue

        # Skip detect-type hitboxes
        if node.get("hitbox_type") == 6:
            continue

        props: dict[str, int | float | str | bool] = {}
        for key in EXTRACT_PROPS:
            if key in node:
                props[key] = node[key]
            elif key in HITBOX_DEFAULTS:
                props[key] = HITBOX_DEFAULTS[key]

        hitboxes.append(HitboxData(
            node_name=name,
            move_state="projectile",
            props=props,
        ))

    return hitboxes


def summarize_move_hitboxes(
    move_state: str, hitboxes: list[HitboxData]
) -> MoveHitboxSummary:
    """Compute a summary of hitbox coverage for a move from its individual hitboxes."""
    # Track the union bounding box across all hitboxes
    h_mins: list[int] = []
    h_maxs: list[int] = []
    v_tops: list[int] = []
    v_bottoms: list[int] = []

    all_hits_aerial = False
    all_hits_grounded = False
    min_startup = 999
    max_active_end = 0
    total_active = 0
    max_damage = 0
    total_damage = 0
    best_hitstun = 0
    best_combo_hitstun = 0
    max_knockback = 0.0
    any_knockdown = False
    best_plus = -999
    hit_heights: set[str] = set()
    raw_hitboxes: list[dict] = []

    for hb in hitboxes:
        p = hb.props
        # Skip Detect-type hitboxes (hitbox_type enum: 6 = Detect)
        if p.get("hitbox_type") == 6:
            continue

        x = int(p.get("x", 0))
        y = int(p.get("y", 0))
        w = abs(int(p.get("width", 20)))  # Hitbox.gd init() normalizes negative
        h = abs(int(p.get("height", 20)))  # Hitbox.gd init() normalizes negative

        # Horizontal range (facing right): hitbox spans from (x - w) to (x + w)
        # But we only care about the forward-facing portion for "reach"
        # x is offset from character center in the facing direction
        h_min = max(
            0, x - w
        )  # clamp negative to 0 (behind character doesn't count as reach)
        h_max = x + w

        h_mins.append(h_min)
        h_maxs.append(h_max)

        # Vertical range: y - h to y + h (negative y = up)
        v_tops.append(y - h)
        v_bottoms.append(y + h)

        # Hit height
        hh = int(p.get("hit_height", 1))
        hit_heights.add(HIT_HEIGHT_MAP.get(hh, "mid"))

        # Target types
        if p.get("hits_vs_aerial", True):
            all_hits_aerial = True
        if p.get("hits_vs_grounded", True) or p.get("hits_vs_standing", True):
            all_hits_grounded = True

        # Frame data
        st = int(p.get("start_tick", 0))
        at = int(p.get("active_ticks", 5))
        min_startup = min(min_startup, st)
        max_active_end = max(max_active_end, st + at)
        total_active += at

        # Damage
        dmg = int(p.get("damage", 0))
        max_damage = max(max_damage, dmg)
        total_damage += dmg

        # Hitstun
        hs = int(p.get("hitstun_ticks", 30))
        chs = int(p.get("combo_hitstun_ticks", -1))
        if chs == -1:
            chs = hs
        best_hitstun = max(best_hitstun, hs)
        best_combo_hitstun = max(best_combo_hitstun, chs)

        # Knockback
        kb_str = p.get("knockback", "10.0")
        try:
            kb = float(kb_str) if isinstance(kb_str, str) else float(kb_str)
        except (ValueError, TypeError):
            kb = 10.0
        max_knockback = max(max_knockback, kb)

        if p.get("knockdown", False):
            any_knockdown = True

        pf = int(p.get("plus_frames", 0))
        best_plus = max(best_plus, pf)

        raw_hitboxes.append(
            {
                "name": hb.node_name,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "damage": dmg,
                "hitstun_ticks": hs,
                "combo_hitstun_ticks": chs,
                "start_tick": st,
                "active_ticks": at,
                "hit_height": HIT_HEIGHT_MAP.get(hh, "mid"),
                "hits_aerial": bool(p.get("hits_vs_aerial", True)),
                "hits_grounded": bool(
                    p.get("hits_vs_grounded", True) or p.get("hits_vs_standing", True)
                ),
            }
        )

    if not h_maxs:
        # No valid hitboxes found
        return MoveHitboxSummary(
            move_state=move_state,
            h_range_min=0,
            h_range_max=0,
            v_top=0,
            v_bottom=0,
            hit_height="none",
            hits_aerial=False,
            hits_grounded=False,
            startup_ticks=0,
            active_ticks=0,
            total_active_ticks=0,
            damage=0,
            total_damage=0,
            hitstun_ticks=0,
            combo_hitstun_ticks=0,
            knockback=0.0,
            knockdown=False,
            plus_frames=0,
            hitboxes=[],
        )

    # Determine hit_height category
    if len(hit_heights) >= 2 or "mid" in hit_heights:
        combined_hit_height = "mid"  # mid hits everything
    elif hit_heights == {"high"}:
        combined_hit_height = "high"
    elif hit_heights == {"low"}:
        combined_hit_height = "low"
    else:
        combined_hit_height = "mid"

    return MoveHitboxSummary(
        move_state=move_state,
        h_range_min=min(h_mins),
        h_range_max=max(h_maxs),
        v_top=min(v_tops),
        v_bottom=max(v_bottoms),
        hit_height=combined_hit_height,
        hits_aerial=all_hits_aerial,
        hits_grounded=all_hits_grounded,
        startup_ticks=min_startup,
        active_ticks=max_active_end - min_startup,
        total_active_ticks=total_active,
        damage=max_damage,
        total_damage=total_damage,
        hitstun_ticks=best_hitstun,
        combo_hitstun_ticks=best_combo_hitstun,
        knockback=max_knockback,
        knockdown=any_knockdown,
        plus_frames=best_plus if best_plus > -999 else 0,
        hitboxes=raw_hitboxes,
    )


def extract_all_characters() -> dict[str, dict[str, MoveHitboxSummary]]:
    """Extract hitbox summaries for all characters + universal moves."""
    result: dict[str, dict[str, MoveHitboxSummary]] = {}

    # Universal moves from BaseChar.tscn
    base_path = DECOMPILE_ROOT / BASE_CHAR_TSCN
    if base_path.exists():
        move_hitboxes = extract_hitboxes_from_tscn(base_path)
        state_data = extract_state_data_from_tscn(base_path)
        summaries = {}
        for move_state, hitboxes in move_hitboxes.items():
            summary = summarize_move_hitboxes(move_state, hitboxes)
            sd = state_data.get(move_state)
            if sd:
                summary.anim_length = sd.anim_length
                summary.iasa_at = sd.iasa_at
                summary.move_type = sd.move_type
                summary.title = sd.title
                summary.selectable = sd.selectable
            summaries[move_state] = summary
        result["_universal"] = summaries

    # Per-character moves
    for char_name, tscn_rel_path in CHARACTER_TSCN_FILES.items():
        tscn_path = DECOMPILE_ROOT / tscn_rel_path
        if not tscn_path.exists():
            print(f"WARNING: {tscn_path} not found, skipping {char_name}")
            continue

        move_hitboxes = extract_hitboxes_from_tscn(tscn_path)
        state_data = extract_state_data_from_tscn(tscn_path)
        summaries = {}

        # 1. Process character .tscn hitboxes
        for move_state, hitboxes in move_hitboxes.items():
            summary = summarize_move_hitboxes(move_state, hitboxes)
            summary.source = "character"
            sd = state_data.get(move_state)
            if sd:
                summary.anim_length = sd.anim_length
                summary.iasa_at = sd.iasa_at
                summary.move_type = sd.move_type
                summary.title = sd.title
                summary.selectable = sd.selectable
            summaries[move_state] = summary

        # 2. Process projectile hitboxes for moves that spawn projectiles
        for (proj_char, proj_move), proj_paths in PROJECTILE_MOVE_MAP.items():
            if proj_char != char_name:
                continue
            if proj_move in summaries and summaries[proj_move].damage > 0:
                # Already has hitbox data from character .tscn — keep it,
                # but merge projectile data if it has higher damage/range
                continue

            all_proj_hitboxes: list[HitboxData] = []
            for proj_rel in proj_paths:
                proj_path = DECOMPILE_ROOT / proj_rel
                all_proj_hitboxes.extend(extract_hitboxes_from_projectile(proj_path))

            if all_proj_hitboxes:
                summary = summarize_move_hitboxes(proj_move, all_proj_hitboxes)
                summary.source = "projectile"
                sd = state_data.get(proj_move)
                if sd:
                    summary.anim_length = sd.anim_length
                    summary.iasa_at = sd.iasa_at
                    summary.move_type = sd.move_type
                    summary.title = sd.title
                    summary.selectable = sd.selectable
                summaries[proj_move] = summary

        # 3. Add stub entries for known no-hitbox moves with state data
        for (nh_char, nh_move), reason in NO_HITBOX_MOVES.items():
            if nh_char != char_name:
                continue
            if nh_move in summaries:
                continue
            sd = state_data.get(nh_move)
            summary = MoveHitboxSummary(
                move_state=nh_move,
                h_range_min=0,
                h_range_max=0,
                v_top=0,
                v_bottom=0,
                hit_height="none",
                hits_aerial=False,
                hits_grounded=False,
                startup_ticks=0,
                active_ticks=0,
                total_active_ticks=0,
                damage=0,
                total_damage=0,
                hitstun_ticks=0,
                combo_hitstun_ticks=0,
                knockback=0.0,
                knockdown=False,
                plus_frames=0,
                hitboxes=[],
                source="no_hitbox",
                source_note=reason,
            )
            if sd:
                summary.anim_length = sd.anim_length
                summary.iasa_at = sd.iasa_at
                summary.move_type = sd.move_type
                summary.title = sd.title
                summary.selectable = sd.selectable
            summaries[nh_move] = summary

        # 4. Apply programmatic overrides for dynamically-positioned hitboxes
        for (ov_char, ov_move), override in PROGRAMMATIC_OVERRIDES.items():
            if ov_char != char_name:
                continue
            if ov_move not in summaries:
                continue
            s = summaries[ov_move]
            for key, val in override.items():
                if hasattr(s, key):
                    setattr(s, key, val)
            if s.source == "character":
                s.source = "override"

        result[char_name] = summaries

    return result


def summaries_to_json(
    all_data: dict[str, dict[str, MoveHitboxSummary]],
    *,
    include_non_selectable: bool = False,
) -> dict:
    """Convert extraction results to a JSON-serializable dict."""
    output: dict = {}
    for char_name, moves in all_data.items():
        char_dict: dict = {}
        for move_name, summary in sorted(moves.items()):
            if not include_non_selectable and not summary.selectable:
                continue
            entry: dict = {
                "h_range_min": summary.h_range_min,
                "h_range_max": summary.h_range_max,
                "v_top": summary.v_top,
                "v_bottom": summary.v_bottom,
                "hit_height": summary.hit_height,
                "hits_aerial": summary.hits_aerial,
                "hits_grounded": summary.hits_grounded,
                "startup_ticks": summary.startup_ticks,
                "active_ticks": summary.active_ticks,
                "damage": summary.damage,
                "total_damage": summary.total_damage,
                "hitstun_ticks": summary.hitstun_ticks,
                "combo_hitstun_ticks": summary.combo_hitstun_ticks,
                "knockback": summary.knockback,
                "knockdown": summary.knockdown,
                "plus_frames": summary.plus_frames,
                "source": summary.source,
                "hitboxes": summary.hitboxes,
            }
            if summary.anim_length is not None:
                entry["anim_length"] = summary.anim_length
            if summary.iasa_at is not None:
                entry["iasa_at"] = summary.iasa_at
            if summary.move_type is not None:
                entry["move_type"] = summary.move_type
            if summary.title is not None:
                entry["title"] = summary.title
            if summary.source_note:
                entry["source_note"] = summary.source_note
            char_dict[move_name] = entry
        output[char_name] = char_dict
    return output


def enrich_move_catalog(
    all_data: dict[str, dict[str, MoveHitboxSummary]],
) -> dict:
    """Merge extracted hitbox data into move_catalog.json and return the enriched catalog."""
    catalog_path = (
        Path(__file__).resolve().parent.parent / "prompts" / "move_catalog.json"
    )
    catalog: dict = json.loads(catalog_path.read_text())

    enriched_count = 0
    missing_in_catalog: list[str] = []

    for char_name, moves in all_data.items():
        if char_name not in catalog:
            continue
        char_catalog = catalog[char_name]

        for move_name, summary in moves.items():
            # Skip non-selectable states
            if not summary.selectable:
                continue

            if move_name not in char_catalog:
                missing_in_catalog.append(f"{char_name}/{move_name}")
                continue

            entry = char_catalog[move_name]

            # Update h_reach — use the larger of existing or extracted.
            # Programmatic overrides already set correct h_range_max for projectiles.
            existing_h_reach = entry.get("h_reach")
            if summary.h_range_max > 0:
                if existing_h_reach is None or summary.h_range_max > existing_h_reach:
                    entry["h_reach"] = summary.h_range_max

            # Add minimum range (deadzone) — only if > 0
            if summary.h_range_min > 0:
                entry["h_range_min"] = summary.h_range_min

            # Vertical range
            if summary.v_top != 0 or summary.v_bottom != 0:
                entry["v_range"] = [summary.v_top, summary.v_bottom]

            # Target restrictions — only add if NOT hitting a category
            if not summary.hits_grounded:
                entry["hits_grounded"] = False
            elif "hits_grounded" in entry and entry["hits_grounded"] is False:
                del entry["hits_grounded"]

            if not summary.hits_aerial:
                entry["hits_aerial"] = False
            elif "hits_aerial" in entry and entry["hits_aerial"] is False:
                del entry["hits_aerial"]

            # Frame data
            if summary.startup_ticks > 0:
                entry["startup_ticks"] = summary.startup_ticks
            if summary.active_ticks > 0:
                entry["active_ticks"] = summary.active_ticks

            # State-level frame data
            if summary.anim_length is not None:
                entry["anim_length"] = summary.anim_length
            if summary.iasa_at is not None:
                entry["iasa_at"] = summary.iasa_at

            # Damage
            if summary.damage > 0:
                entry["total_damage"] = summary.damage
            if summary.total_damage > summary.damage:
                entry["multi_hit_total_damage"] = summary.total_damage

            # Hitstun — the key new data for combo reasoning
            if summary.hitstun_ticks > 0:
                entry["hitstun_ticks"] = summary.hitstun_ticks
            if (
                summary.combo_hitstun_ticks > 0
                and summary.combo_hitstun_ticks != summary.hitstun_ticks
            ):
                entry["combo_hitstun_ticks"] = summary.combo_hitstun_ticks

            # Knockback
            if summary.knockdown:
                entry["knockdown"] = True

            # Plus frames on block
            if summary.plus_frames != 0:
                entry["plus_frames"] = summary.plus_frames

            enriched_count += 1

    if missing_in_catalog:
        print(f"\nMoves in .tscn but NOT in catalog ({len(missing_in_catalog)}):")
        for m in sorted(missing_in_catalog):
            print(f"  {m}")

    print(f"\nEnriched {enriched_count} move entries in catalog")
    return catalog


def main() -> None:
    import sys

    all_data = extract_all_characters()

    # Print summary
    for char_name, moves in all_data.items():
        selectable_count = sum(1 for s in moves.values() if s.selectable)
        non_selectable_count = sum(1 for s in moves.values() if not s.selectable)
        proj_count = sum(1 for s in moves.values() if s.source == "projectile")
        override_count = sum(1 for s in moves.values() if s.source == "override")
        no_hitbox_count = sum(1 for s in moves.values() if s.source == "no_hitbox")

        print(f"\n{'=' * 72}")
        print(
            f"  {char_name}: {selectable_count} selectable moves "
            f"({non_selectable_count} non-selectable filtered)"
        )
        print(
            f"  Sources: {selectable_count - proj_count - override_count - no_hitbox_count} character, "
            f"{proj_count} projectile, {override_count} override, {no_hitbox_count} no_hitbox"
        )
        print(f"{'=' * 72}")
        for move_name, summary in sorted(moves.items()):
            if not summary.selectable:
                continue
            hb_count = len(summary.hitboxes)
            aerial = "aerial" if summary.hits_aerial else ""
            grounded = "grounded" if summary.hits_grounded else ""
            targets = "+".join(filter(None, [aerial, grounded])) or "none"
            src = f"[{summary.source[:4]}]"
            print(
                f"  {move_name:30s} {src:6s} | "
                f"h_range: {summary.h_range_min:3d}-{summary.h_range_max:3d} | "
                f"startup: {summary.startup_ticks:2d} | "
                f"dmg: {summary.damage:3d} | "
                f"hitstun: {summary.hitstun_ticks:2d} | "
                f"targets: {targets:16s} | "
                f"hitboxes: {hb_count}"
            )

    # Write JSON output (selectable moves only)
    output_path = (
        Path(__file__).resolve().parent.parent / "prompts" / "hitbox_data.json"
    )
    json_data = summaries_to_json(all_data, include_non_selectable=False)
    output_path.write_text(json.dumps(json_data, indent=2) + "\n")
    print(f"\nWrote extracted hitbox data to {output_path}")

    # Enrich move_catalog.json if requested
    if "--enrich" in sys.argv:
        enriched_catalog = enrich_move_catalog(all_data)
        catalog_path = (
            Path(__file__).resolve().parent.parent / "prompts" / "move_catalog.json"
        )
        catalog_path.write_text(json.dumps(enriched_catalog, indent=2) + "\n")
        print(f"Wrote enriched catalog to {catalog_path}")

    if "--json" in sys.argv:
        print(json.dumps(json_data, indent=2))


if __name__ == "__main__":
    main()
