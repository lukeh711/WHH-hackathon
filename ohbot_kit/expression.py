"""The empathy layer: poses, gestures and gaze.

This is the vocabulary you remix. A pose is a held facial expression; a gesture
is a timed sequence that plays out; gaze aims the eyes independently of the head.

    from expression import POSES, GESTURES
    bot.express("sympathetic")     # hold a face
    bot.gesture("slow_nod")        # play a movement, underneath speech
    bot.gaze(x=3, y=7)             # look up and to the left

MOUTH OWNERSHIP -- the one rule that matters:

While the robot is speaking, the lip-sync thread owns TOPLIP and BOTTOMLIP. A
pose that also sets the mouth would fight it and produce mush. So every mouth
value here is applied ONLY when not speaking; robot.py enforces this. Change a
pose's mouth freely, but don't remove that guard.

All positions are 0-10. The library clamps them to each robot's calibrated
range from ohbotData/MotorDefinitionsv21.omd, so you cannot drive a servo past
its limits by choosing a silly number here.
"""

from __future__ import annotations

from collections.abc import Iterable

from ohbot import ohbot

# Motor numbers, named for readability. All eight exist -- note that TOPLIP (4)
# and HEADROLL (7) are real despite being absent from the vendor's docs table.
HEADNOD = ohbot.HEADNOD  # 0  pitch: low = looking down
HEADTURN = ohbot.HEADTURN  # 1  yaw: low = its right, high = its left
EYETURN = ohbot.EYETURN  # 2  eye yaw
LIDBLINK = ohbot.LIDBLINK  # 3  10 = wide open, 0 = shut  (rest is 10)
TOPLIP = ohbot.TOPLIP  # 4  mouth -- owned by lip sync while speaking
BOTTOMLIP = ohbot.BOTTOMLIP  # 5  mouth -- owned by lip sync while speaking
EYETILT = ohbot.EYETILT  # 6  eye pitch: low = looking down
HEADROLL = ohbot.HEADROLL  # 7  tilt: the sympathetic/quizzical head cock

# Motors that lip sync controls during speech.
MOUTH = (TOPLIP, BOTTOMLIP)

# Neutral resting position. LIDBLINK rests at 10 (open) -- resting it at 5
# leaves the robot permanently half-lidded and looking half asleep.
REST = {
    HEADNOD: 5,
    HEADTURN: 5,
    EYETURN: 5,
    LIDBLINK: 10,
    EYETILT: 5,
    HEADROLL: 5,
}

# -- poses ----------------------------------------------------------------
# A held expression. Anything omitted keeps its current position, so poses
# compose with whatever gaze you've set.

POSES = {
    "neutral": {LIDBLINK: 10, HEADNOD: 5, EYETILT: 5, HEADROLL: 5},
    # Lifted head, wide-ish eyes, open mouth suggesting a smile.
    "happy": {LIDBLINK: 9, HEADNOD: 6, EYETILT: 6, HEADROLL: 5, BOTTOMLIP: 7, TOPLIP: 6},
    # Everything drops: head, gaze and lids. The head tilt stops it reading
    # as merely "switched off".
    "sad": {LIDBLINK: 5, HEADNOD: 3, EYETILT: 3, HEADROLL: 6, BOTTOMLIP: 4, TOPLIP: 4},
    "surprised": {LIDBLINK: 10, HEADNOD: 6, EYETILT: 6, HEADROLL: 5, BOTTOMLIP: 9, TOPLIP: 8},
    # Looking up and away is the universal "I'm working on it".
    "thinking": {LIDBLINK: 7, HEADNOD: 6, EYETILT: 8, EYETURN: 7, HEADROLL: 6},
    # The head cock. This is the one that makes people say "aww".
    "curious": {LIDBLINK: 10, HEADNOD: 5, EYETILT: 6, HEADROLL: 8},
    # Softened, slightly lowered, head tilted toward you.
    "sympathetic": {LIDBLINK: 6, HEADNOD: 4, EYETILT: 4, HEADROLL: 7, BOTTOMLIP: 5, TOPLIP: 5},
    "excited": {LIDBLINK: 10, HEADNOD: 7, EYETILT: 7, HEADROLL: 5, BOTTOMLIP: 8, TOPLIP: 7},
    "confused": {LIDBLINK: 8, HEADNOD: 5, EYETILT: 5, HEADROLL: 3, EYETURN: 6},
    # Pulled back and slightly away, eyes wide. Pair with the "shiver" gesture.
    "scared": {LIDBLINK: 10, HEADNOD: 3, EYETILT: 6, HEADROLL: 6, BOTTOMLIP: 7, TOPLIP: 6},
    # Wide grin with a lopsided head tilt -- the tilt (via HEADROLL) is what
    # reads as "silly" rather than plain "happy".
    "silly": {LIDBLINK: 9, HEADNOD: 6, EYETILT: 6, HEADROLL: 8, BOTTOMLIP: 8, TOPLIP: 5},
    # Half-closed eyes and a drooping head, like it's about to nod off.
    "sleepy": {LIDBLINK: 3, HEADNOD: 3, EYETILT: 3, HEADROLL: 5, BOTTOMLIP: 4, TOPLIP: 4},
    # Bigger and dumber than "silly" -- wide grin, no self-awareness, for the
    # broadest laugh lines.
    "goofy": {LIDBLINK: 8, HEADNOD: 6, EYETILT: 6, HEADROLL: 9, BOTTOMLIP: 9, TOPLIP: 4},
    # Half-lidded, small closed-mouth smirk, tilted the OPPOSITE way from
    # silly/goofy -- self-satisfied rather than delighted.
    "smug": {LIDBLINK: 6, HEADNOD: 5, EYETILT: 5, HEADROLL: 2, BOTTOMLIP: 6, TOPLIP: 3},
    # A sideways glance (EYETURN off-centre -- the one thing here that reads
    # as "sneaky") plus a small grin: the "I have a plan" face.
    "mischievous": {LIDBLINK: 5, HEADNOD: 5, EYETILT: 4, HEADROLL: 6, EYETURN: 8, BOTTOMLIP: 6, TOPLIP: 4},
}

# Axes that aim the face at the person. A pose may leave these off-centre for
# character -- `curious` tilts the head to 8 -- but they must be returned to
# centre afterwards or the offset persists into the next utterance and
# compounds until the robot is addressing the wall. robot.recentre() does that.
ORIENTATION = (HEADTURN, HEADROLL, EYETURN)

# -- gestures -------------------------------------------------------------
# Keyframes: (motor, position, speed, hold_seconds). Speed is 0-10, where 10
# is fastest. hold_seconds is how long to wait before the next keyframe.
#
# Gestures run on a thread so they play UNDERNEATH speech -- that simultaneity
# is what separates "expressive robot" from "talking statue".

GESTURES = {
    "nod": [
        (HEADNOD, 3, 8, 0.22),
        (HEADNOD, 6, 8, 0.22),
        (HEADNOD, 3, 8, 0.22),
        (HEADNOD, 5, 6, 0.20),
    ],
    # Deliberately slow and shallow -- a brisk nod reads as agreement,
    # a slow one reads as sympathy.
    "slow_nod": [
        (HEADNOD, 4, 2, 0.60),
        (HEADNOD, 5, 2, 0.60),
        (HEADNOD, 4, 2, 0.60),
        (HEADNOD, 5, 2, 0.40),
    ],
    "shake": [
        (HEADTURN, 3, 8, 0.20),
        (HEADTURN, 7, 8, 0.24),
        (HEADTURN, 3, 8, 0.24),
        (HEADTURN, 5, 6, 0.20),
    ],
    "tilt": [
        (HEADROLL, 8, 3, 0.60),
        (HEADROLL, 5, 3, 0.30),
    ],
    # Glance away, then snap back with wide eyes.
    "double_take": [
        (HEADTURN, 7, 7, 0.30),
        (LIDBLINK, 6, 10, 0.15),
        (HEADTURN, 4, 10, 0.15),
        (LIDBLINK, 10, 10, 0.30),
        (HEADTURN, 5, 6, 0.20),
    ],
    "lean_in": [
        (HEADNOD, 7, 3, 0.50),
        (LIDBLINK, 10, 6, 0.40),
    ],
    "perk_up": [
        (LIDBLINK, 10, 10, 0.10),
        (HEADNOD, 7, 8, 0.25),
        (HEADROLL, 6, 6, 0.25),
        (HEADNOD, 5, 5, 0.15),
        (HEADROLL, 5, 6, 0.15),  # return the roll, or it leaks into the next line
    ],
    # Fast, small, irregular -- a tremble, not a shake. Amplitude stays within
    # one unit of centre: big movements read as "shaking the head no", small
    # rapid ones read as fear. Eyes dart as well as the head, which is what
    # sells it; a head-only tremble looks like a loose servo.
    "shiver": [
        (LIDBLINK, 10, 10, 0.05),
        (HEADROLL, 4, 10, 0.06),
        (EYETURN, 6, 10, 0.05),
        (HEADROLL, 6, 10, 0.06),
        (EYETURN, 4, 10, 0.05),
        (HEADROLL, 4, 10, 0.06),
        (HEADTURN, 6, 10, 0.05),
        (HEADROLL, 6, 10, 0.06),
        (HEADTURN, 4, 10, 0.05),
        (HEADROLL, 4, 10, 0.06),
        (EYETURN, 6, 10, 0.05),
        (HEADROLL, 6, 10, 0.06),
        (EYETURN, 4, 10, 0.05),
        (HEADROLL, 5, 8, 0.08),
        (HEADTURN, 5, 8, 0.06),
        (EYETURN, 5, 8, 0.10),
    ],
    # A flinch: snap back and away, hold, then edge cautiously forward again.
    "recoil": [
        (LIDBLINK, 10, 10, 0.04),
        (HEADNOD, 2, 10, 0.10),
        (HEADTURN, 7, 10, 0.30),
        (HEADROLL, 6, 8, 0.35),
        (HEADTURN, 5, 2, 0.40),
        (HEADNOD, 5, 2, 0.25),
        (HEADROLL, 5, 3, 0.20),
    ],
    "look_away": [
        (EYETURN, 8, 5, 0.35),
        (HEADTURN, 6, 3, 0.50),
        (EYETURN, 5, 4, 0.30),
        (HEADTURN, 5, 3, 0.30),
    ],
    "blink": [
        (LIDBLINK, 0, 10, 0.10),
        (LIDBLINK, 10, 10, 0.05),
    ],
    "double_blink": [
        (LIDBLINK, 0, 10, 0.09),
        (LIDBLINK, 10, 10, 0.09),
        (LIDBLINK, 0, 10, 0.09),
        (LIDBLINK, 10, 10, 0.05),
    ],
    # A shimmy -- quick side-to-side head roll, like shaking off excitement.
    "wiggle": [
        (HEADROLL, 8, 10, 0.10),
        (HEADROLL, 2, 10, 0.10),
        (HEADROLL, 8, 10, 0.10),
        (HEADROLL, 2, 10, 0.10),
        (HEADROLL, 5, 8, 0.12),
    ],
    # A "tee-hee": rapid blinking paired with a little head bounce -- finding
    # something funny, not just agreeing with it (that's "nod").
    "giggle": [
        (HEADNOD, 6, 10, 0.10),
        (LIDBLINK, 5, 10, 0.08),
        (HEADNOD, 4, 10, 0.10),
        (LIDBLINK, 10, 10, 0.08),
        (HEADNOD, 6, 10, 0.10),
        (LIDBLINK, 5, 10, 0.08),
        (HEADNOD, 5, 8, 0.12),
        (LIDBLINK, 10, 10, 0.08),
    ],
}

# What each gesture MEANS. The LLM picks from names alone otherwise, and names
# are ambiguous: given just "shake" it will happily pick it for good news, where
# a head shake reads as "no". Measured effect -- without these descriptions,
# "I finally finished my project!" chose `shake`, and "why is the sky blue?"
# chose `double_take`. Keep a description for every gesture you add.
GESTURE_MEANINGS = {
    "nod": "agreement, understanding, yes",
    "slow_nod": "sympathy, taking something in seriously",
    "shake": "disagreement, no, disbelief at something bad",
    "tilt": "curiosity, mild puzzlement",
    "double_take": "shock at something startling",
    "lean_in": "close interest, wanting to hear more",
    "perk_up": "delight at good news",
    "look_away": "embarrassment, discomfort, thinking to oneself",
    "blink": "a small neutral beat",
    "double_blink": "mild confusion or processing",
    "shiver": "fear, dread, being creeped out",
    "recoil": "alarm or disgust at something unpleasant",
    "wiggle": "playful excitement, showing off, being silly",
    "giggle": "amusement, finding something funny",
}

# Which gestures suit which emotion.
#
# Offering all twelve at once measurably under-uses the vocabulary: over 20
# varied prompts the model chose `double_take` 35% of the time and never once
# chose shiver, recoil, tilt or double_blink -- not even for "I think someone is
# following me" or "That's disgusting!".
#
# Narrowing the choice to gestures that fit the chosen emotion fixes both
# halves of that. It is a smaller decision for a small model, and it makes an
# incoherent pairing (a bouncy gesture over sad news) structurally impossible
# rather than merely discouraged by the prompt.
#
# Every gesture must appear at least once here, or it becomes dead code -- a
# test enforces that.
EMOTION_GESTURES = {
    "neutral": ["blink", "nod", "tilt", "look_away"],
    "happy": ["nod", "perk_up", "lean_in", "blink"],
    "excited": ["perk_up", "double_take", "nod", "lean_in"],
    "sad": ["slow_nod", "look_away", "shake"],
    # No `shake` here, despite "disbelief at something bad" fitting the mood.
    # Sympathetic replies often turn towards encouragement -- "don't worry,
    # I'll help you prepare" -- and delivering that with a head shake reads as
    # "no". Observed on hardware; `tilt` carries the same warmth without the
    # contradiction.
    "sympathetic": ["slow_nod", "lean_in", "tilt"],
    "surprised": ["double_take", "recoil", "perk_up"],
    "scared": ["shiver", "recoil", "look_away"],
    "curious": ["tilt", "lean_in", "double_blink", "blink"],
    "confused": ["double_blink", "tilt", "shake"],
    "thinking": ["look_away", "double_blink", "tilt"],
    "silly": ["wiggle", "giggle", "perk_up", "double_take"],
    "sleepy": ["slow_nod", "blink"],
    "goofy": ["wiggle", "giggle", "perk_up"],
    "smug": ["tilt", "nod"],
    "mischievous": ["wiggle", "look_away", "double_blink"],
}

# Emotions the LLM is allowed to pick. Kept in one place so the JSON schema and
# the pose table can never drift apart.
EMOTIONS = sorted(POSES)
GESTURE_NAMES = sorted(GESTURES)


def gestures_for(emotion: str) -> list[str]:
    """Gestures that suit an emotion. Falls back to everything if unmapped."""
    return EMOTION_GESTURES.get(emotion, GESTURE_NAMES)


def gesture_menu(names: Iterable[str] | None = None) -> str:
    """ "name: meaning" lines for the prompt, so choices are made on meaning."""
    return "\n".join(
        "- {}: {}".format(name, GESTURE_MEANINGS.get(name, "no description"))
        for name in (GESTURE_NAMES if names is None else names)
    )


def gaze_positions(x: float = 5, y: float = 5) -> dict[int, float]:
    """Eye targets for a gaze direction. x: 0 right .. 10 left, y: 0 down .. 10 up."""
    return {EYETURN: x, EYETILT: y}


# Named targets as (x, y), where x is 0 right .. 10 left and y is 0 down .. 10 up.
# "user" is straight ahead -- the position to return to so it addresses the person.
LOOK_DIRECTIONS = {
    "user": (5, 5),
    "ahead": (5, 5),
    "left": (9, 5),
    "right": (1, 5),
    "up": (5, 9),
    "down": (5, 1),
    "up_left": (8, 8),
    "up_right": (2, 8),
    "down_left": (8, 2),
    "down_right": (2, 2),
}

# How far the head follows the eyes. People saccade first and the head catches
# up partway; driving the head the full distance makes the robot look like it
# is tracking a fly rather than glancing at something.
HEAD_FOLLOW = 0.55


def look_targets(
    x: float = 5,
    y: float = 5,
    with_head: bool = True,
    head_follow: float = HEAD_FOLLOW,
) -> tuple[dict[int, float], dict[int, float]]:
    """Motor targets for looking somewhere with eyes, head yaw, pitch and roll.

    Returns (eye_targets, head_targets) so the caller can move the eyes first
    and the head a moment later, which is what makes the movement read as a
    glance rather than a mechanical pan.
    """
    eyes = {EYETURN: x, EYETILT: y}
    if not with_head:
        return eyes, {}

    def follow(value: float) -> float:
        return 5 + (value - 5) * head_follow

    head = {
        HEADTURN: follow(x),
        HEADNOD: follow(y),
        # A slight roll toward the direction of travel. Real heads tip a little
        # when turning to look, and without it the turn feels stiff.
        HEADROLL: 5 + (x - 5) * 0.2,
    }
    return eyes, head
