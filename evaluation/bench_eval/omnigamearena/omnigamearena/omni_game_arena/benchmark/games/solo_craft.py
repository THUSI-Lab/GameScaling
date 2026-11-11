"""SoloCraft GameSpec.

Top-down 3D cookhouse-style arena: pick up parts, optionally process them
at the workbench, then submit them at the order counter for points within a
fixed match timer.

Terminal signals surfaced by the UE5 scene:
  - ``score``       : raw accumulated order points V. Official paper
    score is ``min(V / V_TARGET, 1)`` with default ``V_TARGET=50``.
  - ``done_reason`` : one of ``"timeout"``, ``"max_steps"``, ``"agent"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import GameSpec

# Author default: paper score = V / 50. UE get_score returns raw points.
V_TARGET = 50.0


def normalize_solo_craft_score(score, game_metrics=None):
    """Map a SoloCraft result to the paper [0, 1] scale.

    New runs store already-normalized ``score`` plus ``raw_score``. Old
    Earlier cookhouse runs stored integer order points (1.0 / 2.0)
    without that field. Fractional scores in (0, 1) are left as-is: the
    pre-map-fix episodes that leaked ObstacleRun3D sit in that band and
    must not be divided again.
    """
    if not isinstance(score, (int, float)):
        return score
    gm = game_metrics or {}
    if gm.get("raw_score") is not None:
        return float(score)
    s = float(score)
    if s >= 1:
        return min(s / V_TARGET, 1.0)
    return s


_KEY_BINDINGS = {
    "W": "Move up",
    "A": "Move left",
    "S": "Move down",
    "D": "Move right",
    "F": "Interact (pick up item / submit at order counter / process at workbench / discard)",
}


@dataclass
class SoloCraftSpec(GameSpec):
    name: str = "solo_craft"
    prompt_key: str = "SoloCraft"
    # Must be the maps.yaml key. Empty map skips `open`, and a freshly
    # launched UE5 stays on its boot level (ObstacleRun3D) — every parallel
    # solo_craft run before this was therefore the 3D obstacle course with
    # cookhouse WASD+F bindings. Official cold-start YAML sets this explicitly.
    map: str = "solo_craft"
    default_task: str = (
        "Collect parts scattered around the arena and submit ones that match "
        "the active order at the order counter to maximize score before the "
        "match timer runs out. For high-grade orders, process the matching "
        "part at the workbench before submitting it."
    )
    # Fixed top-down camera, no mouse input.
    mouse_axes: tuple[str, ...] = ()
    key_bindings: dict = field(default_factory=lambda: dict(_KEY_BINDINGS))
    tap_keys: tuple[str, ...] = ("F",)
    chunk_steps: int = 6
    # Pickup / submit / upgrade animations need a moment to settle so the
    # next screenshot reflects the post-interaction inventory state.
    obs_delay: float = 0.5

    def extract_terminal_metrics(self, terminal_info: dict) -> dict:
        info = terminal_info or {}
        raw = info.get("score")
        if raw is None:
            return {"score": None}
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return {"score": raw}
        return {
            "score": min(v / V_TARGET, 1.0),
            "raw_score": v,
        }

    def aggregate_episode_metrics(self, episode_metrics: list[dict]) -> dict:
        game_parts = [(m.get("game") or {}) for m in episode_metrics]
        scores = [g.get("score") for g in game_parts if g.get("score") is not None]
        if not scores:
            return {}
        return {
            "mean_score": round(sum(scores) / len(scores), 4),
        }


SPEC = SoloCraftSpec()
