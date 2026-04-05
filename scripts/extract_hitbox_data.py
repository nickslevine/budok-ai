"""Extract hitbox data from decompiled .tscn files and merge into move_catalog.json.

Parses Godot .tscn scene files for each character, extracts Hitbox node properties
(geometry, hitstun, damage, frame data), and maps them back to their parent move state.

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
        "note": "Slider-controlled range (30-120 center). Deadzone at close range (<18 units).",
    },
}


@dataclass
class HitboxData:
    """Raw hitbox data extracted from a .tscn node."""

    node_name: str
    move_state: str
    props: dict[str, int | float | str | bool] = field(default_factory=dict)


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
    # Raw hitbox list for detailed inspection
    hitboxes: list[dict]


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
        summaries = {}
        for move_state, hitboxes in move_hitboxes.items():
            summaries[move_state] = summarize_move_hitboxes(move_state, hitboxes)
        result["_universal"] = summaries

    # Per-character moves
    for char_name, tscn_rel_path in CHARACTER_TSCN_FILES.items():
        tscn_path = DECOMPILE_ROOT / tscn_rel_path
        if not tscn_path.exists():
            print(f"WARNING: {tscn_path} not found, skipping {char_name}")
            continue
        move_hitboxes = extract_hitboxes_from_tscn(tscn_path)
        summaries = {}
        for move_state, hitboxes in move_hitboxes.items():
            summary = summarize_move_hitboxes(move_state, hitboxes)
            # Apply programmatic overrides from .gd state scripts
            override = PROGRAMMATIC_OVERRIDES.get((char_name, move_state))
            if override:
                for key, val in override.items():
                    if hasattr(summary, key):
                        setattr(summary, key, val)
            summaries[move_state] = summary
        result[char_name] = summaries

    return result


def summaries_to_json(
    all_data: dict[str, dict[str, MoveHitboxSummary]],
) -> dict:
    """Convert extraction results to a JSON-serializable dict."""
    output: dict = {}
    for char_name, moves in all_data.items():
        char_dict: dict = {}
        for move_name, summary in sorted(moves.items()):
            char_dict[move_name] = {
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
                "hitboxes": summary.hitboxes,
            }
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
            if move_name not in char_catalog:
                missing_in_catalog.append(f"{char_name}/{move_name}")
                continue

            entry = char_catalog[move_name]

            # Update h_reach — but only if extracted value is LARGER than existing.
            # Projectile moves (FlameWave, Shoot, etc.) have manually-set h_reach
            # that reflects the projectile's travel distance, not the local hitbox.
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
                # Previously marked false but now true — remove the restriction
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
        print(f"\n{'=' * 60}")
        print(f"  {char_name}: {len(moves)} moves with hitboxes")
        print(f"{'=' * 60}")
        for move_name, summary in sorted(moves.items()):
            hb_count = len(summary.hitboxes)
            aerial = "aerial" if summary.hits_aerial else ""
            grounded = "grounded" if summary.hits_grounded else ""
            targets = "+".join(filter(None, [aerial, grounded])) or "none"
            print(
                f"  {move_name:30s} | "
                f"h_range: {summary.h_range_min:3d}-{summary.h_range_max:3d} | "
                f"v: {summary.v_top:4d} to {summary.v_bottom:3d} | "
                f"startup: {summary.startup_ticks:2d} | "
                f"dmg: {summary.damage:3d} | "
                f"hitstun: {summary.hitstun_ticks:2d} | "
                f"targets: {targets:16s} | "
                f"hitboxes: {hb_count}"
            )

    # Write JSON output
    output_path = (
        Path(__file__).resolve().parent.parent / "prompts" / "hitbox_data.json"
    )
    json_data = summaries_to_json(all_data)
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
