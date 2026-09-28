"""Invariants for the pose and gesture vocabulary.

Every test here corresponds to a bug that actually shipped. Adding a pose or a
gesture is the most common change a hackathon participant will make, and these
are the mistakes that change invites.
"""

from __future__ import annotations

import pytest

from ohbot_kit import expression as ex

ALL_MOTORS = {
    ex.HEADNOD,
    ex.HEADTURN,
    ex.EYETURN,
    ex.LIDBLINK,
    ex.TOPLIP,
    ex.BOTTOMLIP,
    ex.EYETILT,
    ex.HEADROLL,
}


@pytest.mark.parametrize("name", sorted(ex.GESTURES))
def test_gesture_returns_orientation_to_centre(name: str) -> None:
    """A gesture must leave the robot facing the person.

    REGRESSION: `perk_up` ended with HEADROLL at 6. Because nothing reset it,
    the tilt persisted into the next utterance and compounded until the robot
    was addressing the wall instead of the user.
    """
    final: dict[int, float] = {}
    for motor, pos, _speed, _hold in ex.GESTURES[name]:
        final[motor] = pos

    off_centre = {m: p for m, p in final.items() if m in ex.ORIENTATION and p != 5}
    assert not off_centre, (
        f"gesture {name!r} ends with orientation off-centre: {off_centre}. "
        "End the sequence by returning HEADTURN/HEADROLL/EYETURN to 5."
    )


@pytest.mark.parametrize("name", sorted(ex.GESTURES))
def test_every_gesture_has_a_meaning(name: str) -> None:
    """The LLM chooses gestures by description, not by name.

    REGRESSION: given bare names, the model picked `shake` -- which means "no"
    -- for "I finally finished my project!". Descriptions fixed it, so a
    gesture without one silently reintroduces the bug.
    """
    assert name in ex.GESTURE_MEANINGS, (
        f"gesture {name!r} has no entry in GESTURE_MEANINGS. Without a "
        "description the model picks it on the name alone."
    )
    assert ex.GESTURE_MEANINGS[name].strip()


@pytest.mark.parametrize("name", sorted(ex.POSES))
def test_pose_values_are_valid(name: str) -> None:
    for motor, pos in ex.POSES[name].items():
        assert motor in ALL_MOTORS, f"pose {name!r} uses unknown motor {motor}"
        assert 0 <= pos <= 10, f"pose {name!r} sets motor {motor} to {pos}, outside 0-10"


@pytest.mark.parametrize("name", sorted(ex.GESTURES))
def test_gesture_frames_are_valid(name: str) -> None:
    for motor, pos, speed, hold in ex.GESTURES[name]:
        assert motor in ALL_MOTORS, f"gesture {name!r} uses unknown motor {motor}"
        assert 0 <= pos <= 10
        assert 0 <= speed <= 10
        assert 0 < hold < 2, f"gesture {name!r} holds for {hold}s, which will feel broken"


def test_exported_names_match_the_tables() -> None:
    """EMOTIONS and GESTURE_NAMES become the LLM's JSON schema enum. If they
    drift from the tables, the model can pick something that doesn't exist."""
    assert sorted(ex.POSES) == ex.EMOTIONS
    assert sorted(ex.GESTURES) == ex.GESTURE_NAMES


def test_neutral_pose_exists_and_is_centred() -> None:
    assert "neutral" in ex.POSES
    for motor, pos in ex.POSES["neutral"].items():
        if motor in ex.ORIENTATION:
            assert pos == 5


def test_lids_rest_open() -> None:
    """REGRESSION: REST had LIDBLINK at 5 (half-closed), while the robot's own
    calibration file rests it at 10. The robot sat looking half asleep."""
    assert ex.REST[ex.LIDBLINK] == 10


def test_gesture_menu_lists_every_gesture_with_its_meaning() -> None:
    menu = ex.gesture_menu()
    for name in ex.GESTURE_NAMES:
        assert name in menu
        assert ex.GESTURE_MEANINGS[name] in menu


class TestLookTargets:
    def test_eyes_go_all_the_way(self) -> None:
        eyes, _head = ex.look_targets(x=9, y=1)
        assert eyes[ex.EYETURN] == 9
        assert eyes[ex.EYETILT] == 1

    def test_head_follows_only_partway(self) -> None:
        """The head lagging behind the eyes is what makes it read as a glance
        rather than a security camera pan."""
        eyes, head = ex.look_targets(x=10, y=10)
        assert head[ex.HEADTURN] < eyes[ex.EYETURN]
        assert head[ex.HEADNOD] < eyes[ex.EYETILT]
        assert head[ex.HEADTURN] > 5  # but it does move

    def test_head_can_be_left_out(self) -> None:
        eyes, head = ex.look_targets(x=9, y=5, with_head=False)
        assert eyes and head == {}

    def test_centre_stays_centred(self) -> None:
        eyes, head = ex.look_targets(x=5, y=5)
        assert eyes[ex.EYETURN] == 5
        assert head[ex.HEADTURN] == 5
        assert head[ex.HEADROLL] == 5

    def test_user_direction_is_straight_ahead(self) -> None:
        assert ex.LOOK_DIRECTIONS["user"] == (5, 5)
        assert ex.LOOK_DIRECTIONS["ahead"] == (5, 5)

    @pytest.mark.parametrize("name", sorted(ex.LOOK_DIRECTIONS))
    def test_directions_are_in_range(self, name: str) -> None:
        x, y = ex.LOOK_DIRECTIONS[name]
        assert 0 <= x <= 10 and 0 <= y <= 10


class TestEmotionGestureAffinity:
    """The affinity map narrows the LLM's gesture choice to what suits the
    emotion. Measured effect: gestures never chosen fell from 4 to 3, the most
    common gesture fell from 35% to 20% of picks, and incoherent pairings
    became impossible rather than merely discouraged."""

    @pytest.mark.parametrize("emotion", sorted(ex.POSES))
    def test_every_emotion_has_suitable_gestures(self, emotion: str) -> None:
        suited = ex.gestures_for(emotion)
        assert suited, f"emotion {emotion!r} has no gestures to choose from"
        for gesture in suited:
            assert gesture in ex.GESTURES, f"{emotion!r} lists unknown gesture {gesture!r}"

    def test_every_gesture_is_reachable(self) -> None:
        """A gesture in no shortlist can never be chosen, so it is dead code
        that still costs a slot in the prompt."""
        reachable = {g for gs in ex.EMOTION_GESTURES.values() for g in gs}
        unreachable = set(ex.GESTURE_NAMES) - reachable
        assert not unreachable, (
            f"gestures no emotion can reach: {sorted(unreachable)}. "
            "Add each to at least one entry in EMOTION_GESTURES."
        )

    def test_sad_emotions_exclude_bouncy_gestures(self) -> None:
        """Structural version of the tone check: a cheerful movement under bad
        news undercuts the words more than a mismatched face does."""
        for emotion in ("sad", "sympathetic"):
            assert "perk_up" not in ex.gestures_for(emotion)
            assert "double_take" not in ex.gestures_for(emotion)

    def test_happy_emotions_exclude_head_shake(self) -> None:
        """A head shake means "no". REGRESSION: it was chosen for
        "I finally finished my project!"."""
        for emotion in ("happy", "excited"):
            assert "shake" not in ex.gestures_for(emotion)

    def test_fear_reaches_the_fear_gestures(self) -> None:
        assert "shiver" in ex.gestures_for("scared")

    def test_unknown_emotion_falls_back_to_everything(self) -> None:
        assert ex.gestures_for("not_an_emotion") == ex.GESTURE_NAMES

    def test_sympathetic_excludes_head_shake(self) -> None:
        """REGRESSION: observed on hardware -- "But don't worry, I'll help you
        prepare" was delivered with a head shake, i.e. encouragement plus "no".
        Sympathetic replies often turn towards encouragement, so the gesture
        must not contradict that."""
        assert "shake" not in ex.gestures_for("sympathetic")
