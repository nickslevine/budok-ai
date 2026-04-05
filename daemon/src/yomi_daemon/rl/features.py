"""Feature extraction from protocol observations for RL.

Converts the rich, nested Observation/FighterObservation into flat
numeric features suitable for reward computation, logging, and
(eventually) non-LLM policy networks.
"""

from __future__ import annotations

from dataclasses import dataclass

from yomi_daemon.protocol import FighterObservation, Observation


@dataclass(frozen=True, slots=True)
class SituationFeatures:
    """Flat numeric summary of a game situation from one player's perspective."""

    # HP (normalized to [0, 1])
    my_hp_pct: float
    opp_hp_pct: float
    hp_advantage: float  # my_hp_pct - opp_hp_pct

    # Resources
    my_meter: int
    opp_meter: int
    my_burst: int
    opp_burst: int

    # Spatial
    distance_x: float  # absolute horizontal distance
    distance_y: float  # vertical difference (positive = I'm higher)
    my_x_normalized: float  # position relative to stage center [-1, 1]
    opp_x_normalized: float

    # State
    my_grounded: bool
    opp_grounded: bool
    my_in_hitstun: bool
    opp_in_hitstun: bool
    my_in_blockstun: bool
    opp_in_blockstun: bool
    my_combo_count: int
    opp_combo_count: int

    # Frame state
    my_interruptable: bool
    opp_interruptable: bool
    my_can_feint: bool

    @staticmethod
    def extract(obs: Observation, player_id: str) -> SituationFeatures:
        """Extract features from an observation for the given player."""
        me, opp = _resolve_fighters(obs, player_id)

        stage_width = 1100.0  # default
        raw_width = obs.stage.get("width")
        if isinstance(raw_width, int | float):
            stage_width = float(raw_width)

        half_width = stage_width / 2.0 if stage_width > 0 else 1.0

        my_hp_pct = me.hp / me.max_hp if me.max_hp > 0 else 0.0
        opp_hp_pct = opp.hp / opp.max_hp if opp.max_hp > 0 else 0.0

        return SituationFeatures(
            my_hp_pct=my_hp_pct,
            opp_hp_pct=opp_hp_pct,
            hp_advantage=my_hp_pct - opp_hp_pct,
            my_meter=me.meter,
            opp_meter=opp.meter,
            my_burst=me.burst,
            opp_burst=opp.burst,
            distance_x=abs(me.position.x - opp.position.x),
            distance_y=me.position.y - opp.position.y,
            my_x_normalized=me.position.x / half_width,
            opp_x_normalized=opp.position.x / half_width,
            my_grounded=me.grounded,
            opp_grounded=opp.grounded,
            my_in_hitstun=_is_hitstun(me),
            opp_in_hitstun=_is_hitstun(opp),
            my_in_blockstun=me.blockstun > 0,
            opp_in_blockstun=opp.blockstun > 0,
            my_combo_count=me.combo_count,
            opp_combo_count=opp.combo_count,
            my_interruptable=me.state_interruptable,
            opp_interruptable=opp.state_interruptable,
            my_can_feint=me.can_feint,
        )


def distance_label(distance_x: float) -> str:
    """Categorize horizontal distance into a human-readable range label."""
    if distance_x < 100:
        return "point_blank"
    if distance_x < 200:
        return "close"
    if distance_x < 400:
        return "mid"
    if distance_x < 600:
        return "far"
    return "full_screen"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_fighters(
    obs: Observation, player_id: str
) -> tuple[FighterObservation, FighterObservation]:
    fighters = obs.fighters
    if fighters[0].id == player_id:
        return fighters[0], fighters[1]
    return fighters[1], fighters[0]


def _is_hitstun(fighter: FighterObservation) -> bool:
    state = fighter.current_state.lower()
    return "hitstun" in state or "stun" in state
