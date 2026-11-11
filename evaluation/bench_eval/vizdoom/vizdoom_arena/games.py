"""The four ViZDoom maps of the Game-TARS FPS benchmark (arXiv 2510.23691 Fig. 8).

Every engine setting lives here rather than in a `.cfg` file. ViZDoom's
`load_config` resolves `doom_scenario_path` relative to the *current working
directory*, so a cfg-driven harness silently loads the wrong WAD (or none) when
a worker is started from another directory -- the episode then still runs, on
the default map, and reports a plausible score. Declaring the spec in Python
keeps the WAD path absolute and every knob visible in one place.

Score definitions (`GameSpec.score`)
------------------------------------
`total_reward`  ViZDoom's own accumulated reward. Both `defend_the_*` WADs carry
                ACS scripts granting +1 per kill, and we set `death_penalty=1`,
                so this is "kills minus one if you died" -- the number the
                ViZDoom scenario README calls the episode reward.
`killcount`     `D3_battle` / `D4_battle2` declare `living_reward = 0` and
                `death_penalty = 0`. Direct Future Prediction (which introduced
                both maps) reports **frags**, so that is what we score.

                An earlier version of this note claimed the battle WADs carry no
                reward script and that `get_total_reward()` is identically 0.0 on
                them. That is wrong: their ACS `script 5` (the per-kill hook) does
                `reward = reward + 1.0` on ACS global 0, which is exactly the
                variable ViZDoom surfaces as the episode reward. Measured over
                1682 `battle1` and 1676 `battle2` episodes, `total_reward` equals
                `killcount` in **every** one. So the two score modes coincide here
                and the recorded numbers were never affected -- but it does mean a
                paper reporting "average episode reward" on these maps (GameWAM,
                arXiv 2608.26200 Fig. 4) is reporting the same quantity we call
                killcount, and the metric is comparable even though the protocol
                is not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

SCENARIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")

TICS_PER_SECOND = 35  # Doom runs at a fixed 35 tics/s


@dataclass(frozen=True)
class GameSpec:
    name: str
    wad: str
    goal: str                       # one line, goes into the guided system prompt
    tutorial: tuple[str, ...]       # numbered "Game Tutorials" bullets
    # -- `oga` prompt style (the default) ---------------------------------
    # These three mirror what OmniGameArena's factory reads off a UE5 GameSpec,
    # so the assembled system prompt can be byte-identical to the one every
    # previously recorded OmniGameArena result was measured with.
    prompt_key: str = ""            # the {name} in "currently playing {name}"
    rules: str = ""                 # the "**Game Rules**" block
    key_bindings: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    score: str = "total_reward"     # total_reward | killcount
    doom_map: str = "map01"
    # `doom_skill` has only THREE distinct settings on these four maps, measured
    # by docker/vzd_probe_difficulty.py: skills 2, 3 and 4 come out byte-identical
    # on every column (tics survived, kills, damage taken). Their monsters are
    # spawned by script rather than by per-skill thing flags, so the monster set
    # does not respond to skill at all, and the only Doom rules left that bite
    # are global: skill 1 halves incoming damage, skill 5 raises monster attack
    # rate. Do not read 1..5 as a five-point difficulty axis here.
    #
    # 1 since 2026-09-08 (was battle=2 / defend=3, which were equivalent). This
    # roughly doubles survival, and RAISES THE FLOOR: a blind spin-and-fire
    # script scores 9.7 kills on defend_the_center at skill 1, above the best
    # checkpoint measured at the old skill (9.40). Read every skill-1 score
    # against the scripted baselines (e.g. the 9.7 above), not against zero.
    doom_skill: int = 1
    living_reward: float = 0.0
    death_penalty: float = 0.0
    episode_timeout: int = 2100     # tics; 2100 / 35 = 60 s of game time
    episode_start_time: int = 0     # tics skipped before the agent acts
    max_steps: int = 350            # 350 * 6 tics = 2100, i.e. the natural timeout
    # Weapon slots the map actually gives the player, for the prompt's key list.
    weapon_slots: tuple[int, ...] = (1, 2)
    extra_game_variables: tuple[str, ...] = field(default_factory=tuple)

    @property
    def wad_path(self) -> str:
        return os.path.join(SCENARIO_DIR, self.wad)


_COMBAT_TUTORIAL = (
    "Aim with the mouse: move the mouse right to turn right, left to turn left. "
    "The crosshair is the centre of the screen; a target is hit when it is under the crosshair.",
    "Hold LB (left mouse button) to fire. Firing costs ammo -- the AMMO counter is "
    "on the left of the HUD, HEALTH next to it. When HEALTH reaches 0 the episode ends.",
    "Strafe with A / D and move with W / S to dodge incoming fireballs; standing still "
    "in the open is how most episodes end early.",
    "Keys are held for the 33ms of their group, so put the same key in several "
    "consecutive groups to hold it down (e.g. `LB ; LB ; LB ; LB ; LB ; LB` fires "
    "for the whole 200ms).",
)

# -- `oga` style Game Rules / Common Keys ---------------------------------
# Written in the shape of ../omnigamearena/.../prompts/games/*.txt (a lead
# paragraph, "## What you see on screen", "## Rules"), because the assembled
# prompt has to be structurally the same document those runs used. Content is
# the same information the `guided` goal+tutorial carried; only the layout and
# the surrounding skeleton changed.
_SCREEN_COMMON = """## What you see on screen
- First-person view.
- The crosshair at screen centre marks where shots land.
- The status bar along the bottom shows HEALTH on the left and AMMO on the right."""

_RULES_BATTLE = """You are playing Doom, a first-person shooter, in an arena full of monsters. \
Survive and kill as many monsters as you can.

{screen}
- Monsters keep arriving for the whole episode; some throw fireballs from range.

## Rules
- Score is the number of monsters you kill in the episode.
- Firing costs ammo, and the episode ends as soon as your health reaches 0, so \
dodging matters as much as shooting: standing still in the open is how most \
episodes end early.
- Health packs and ammo boxes lie around the map; walk over one to pick it up. \
Running out of either is fatal, so collect them while fighting.
- A target is hit when it is under the crosshair, so turn with the mouse until \
the monster is centred before firing.
{extra}"""

_RULES_DEFEND = """You are playing Doom, a first-person shooter. Monsters arrive in waves and \
walk towards you; shoot them before they reach you.

{screen}

## Rules
- You score +1 for every monster you kill and -1 if you die, so the episode \
reward is your kills minus one if you died.
- {position}
- Firing costs ammo and there is no more to pick up, so fire when a monster is \
actually under the crosshair.
- The episode ends when your health reaches 0 or when the time limit is reached.
{extra}"""

_KEYS_COMMON: tuple[tuple[str, str], ...] = (
    ("W", "Move forward"),
    ("A", "Strafe left"),
    ("S", "Move backward"),
    ("D", "Strafe right"),
    ("LMB", "Fire weapon"),
    ("Ctrl", "Fire weapon (keyboard alternative)"),
    ("Shift", "Run"),
    ("Space", "Open doors / use switches"),
)


def _weapon_keys(slots: tuple[int, ...]) -> tuple[tuple[str, str], ...]:
    words = {1: "one", 2: "two", 3: "three", 4: "four",
             5: "five", 6: "six", 7: "seven"}
    return tuple((words[i], f"Select weapon slot {i}") for i in slots if i in words)


GAMES: dict[str, GameSpec] = {
    # -- Battle-1 / Battle-2: DFP's D3/D4. Open arenas, the player must survive
    #    while collecting health/ammo and killing whatever walks in.
    "battle1": GameSpec(
        name="battle1",
        wad="D3_battle.wad",
        score="killcount",
        doom_skill=1,
        living_reward=0.0,
        death_penalty=0.0,
        goal="Survive in the arena and kill as many monsters as possible.",
        tutorial=_COMBAT_TUTORIAL + (
            "Health packs and ammo boxes are scattered around the arena; walk over one "
            "to pick it up. Running out of either is fatal, so collect them while fighting.",
        ),
        prompt_key="Doom Battle-1",
        rules=_RULES_BATTLE.format(
            screen=_SCREEN_COMMON,
            extra="- The map is one large open arena, so monsters can close in from "
                  "any direction.",
        ),
        key_bindings=_KEYS_COMMON + _weapon_keys((1, 2, 3)),
        weapon_slots=(1, 2, 3),
        extra_game_variables=("KILLCOUNT", "DEATHCOUNT", "HEALTH", "AMMO2", "USER2"),
    ),
    "battle2": GameSpec(
        name="battle2",
        wad="D4_battle2.wad",
        score="killcount",
        doom_skill=1,
        living_reward=0.0,
        death_penalty=0.0,
        goal="Survive in the maze arena and kill as many monsters as possible.",
        tutorial=_COMBAT_TUTORIAL + (
            "This map is a maze rather than one open room: corridors and corners mean "
            "monsters appear at short range, and you can break line of sight to heal.",
            "Health packs and ammo boxes are scattered through the maze; walk over one "
            "to pick it up.",
        ),
        prompt_key="Doom Battle-2",
        rules=_RULES_BATTLE.format(
            screen=_SCREEN_COMMON,
            extra="- The map is a maze rather than one open room: corridors and "
                  "corners mean monsters appear at short range, and you can break "
                  "line of sight to survive.",
        ),
        key_bindings=_KEYS_COMMON + _weapon_keys((1, 2, 3)),
        weapon_slots=(1, 2, 3),
        extra_game_variables=("KILLCOUNT", "DEATHCOUNT", "HEALTH", "AMMO2", "USER2"),
    ),
    # -- Defend the Center / Line: ViZDoom's own scenarios. The player is fixed in
    #    place (no movement buttons in the original cfg) and only turns and fires.
    "defend_the_center": GameSpec(
        name="defend_the_center",
        wad="defend_the_center.wad",
        score="total_reward",
        doom_skill=1,
        living_reward=0.0,
        death_penalty=1.0,
        episode_start_time=10,      # the gun is being unholstered before this
        goal="You stand in the centre of a circular arena. Monsters spawn around the "
             "wall and walk towards you -- turn to face them and shoot them before "
             "they reach you.",
        tutorial=_COMBAT_TUTORIAL + (
            "You cannot leave the centre of the arena; W/A/S/D barely help here. "
            "Turning to face the next monster and firing is the whole game.",
            "You score +1 for every monster killed and -1 if you die, so ammo "
            "discipline matters: fire when a monster is under the crosshair.",
        ),
        prompt_key="Doom Defend the Center",
        rules=_RULES_DEFEND.format(
            screen=_SCREEN_COMMON,
            position="You stand in the centre of a circular arena and cannot leave "
                     "it: monsters spawn against the wall all around you, W/A/S/D "
                     "barely help, and turning to face the next monster is the whole "
                     "game.",
            extra="- Monsters here reach you on foot and attack in melee, so the "
                  "danger is letting one close the distance while you are facing "
                  "the wrong way.",
        ),
        key_bindings=_KEYS_COMMON + _weapon_keys((1, 2)),
        weapon_slots=(1, 2),
        extra_game_variables=("KILLCOUNT", "DEATHCOUNT", "HEALTH", "AMMO2"),
    ),
    "defend_the_line": GameSpec(
        name="defend_the_line",
        wad="defend_the_line.wad",
        score="total_reward",
        doom_skill=1,
        living_reward=0.0,
        death_penalty=1.0,
        episode_start_time=10,
        # The shipped defend_the_line.cfg has NO episode_timeout, so a cfg-driven
        # run would only end when the agent dies -- unbounded wall clock for a
        # model that happens to survive. Bound it like every other map.
        episode_timeout=2100,
        goal="Monsters spawn on the far side of the room and advance towards you in a "
             "line. Turn to face them and shoot them before they reach you.",
        tutorial=_COMBAT_TUTORIAL + (
            "The monsters here shoot back from range, so keep firing at whatever is "
            "under the crosshair and expect to take damage.",
            "You score +1 for every monster killed and -1 if you die.",
        ),
        prompt_key="Doom Defend the Line",
        rules=_RULES_DEFEND.format(
            screen=_SCREEN_COMMON,
            position="You stand at one end of a rectangular room and the monsters "
                     "advance towards you from the far side, so the fight is a "
                     "shooting gallery: turn onto the nearest one and keep firing.",
            extra="- These monsters shoot back from range, so expect to take damage "
                  "and kill them before they whittle you down.",
        ),
        key_bindings=_KEYS_COMMON + _weapon_keys((1, 2)),
        weapon_slots=(1, 2),
        extra_game_variables=("KILLCOUNT", "DEATHCOUNT", "HEALTH", "AMMO2"),
    ),
}

SUITE = ("battle1", "battle2", "defend_the_center", "defend_the_line")

# Relative per-episode cost, quoted in minutes at ~2 s/step. Only the ratios
# matter -- they feed the longest-processing-time worker split, which is
# unchanged by a constant factor. The absolute numbers are wrong for a thinking
# baseline by an order of magnitude (Qwen3-VL-8B-Thinking measured 22.8 s/step
# mean, 45 s p90, i.e. ~2 h for a 350-step episode), so do not read the planner's
# "est min" column as a schedule.
# battle* usually die well before the timeout; defend_the_* run long.
EST_MINUTES = {"battle1": 6.0, "battle2": 8.0,
               "defend_the_center": 12.0, "defend_the_line": 12.0}


def get_game(name: str) -> GameSpec:
    try:
        return GAMES[name]
    except KeyError:
        raise KeyError(f"unknown map {name!r}; known: {', '.join(sorted(GAMES))}") from None
