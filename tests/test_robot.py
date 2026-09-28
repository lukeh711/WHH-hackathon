"""Robot behaviour, asserted against the recording fake.

These cover the rules that are invisible in the code but obvious on the robot:
who owns the mouth, and whether it ends up facing the person.
"""

from __future__ import annotations

import pytest

from ohbot_kit import expression as ex
from ohbot_kit.robot import Ohbot
from tests.conftest import FakeOhbot


class TestMouthOwnership:
    def test_pose_does_not_drive_the_mouth_while_speaking(
        self, bot: Ohbot, fake_ohbot: FakeOhbot
    ) -> None:
        """While speaking, the lip-sync thread owns TOPLIP and BOTTOMLIP. A
        pose setting them too produces mush, so those values are dropped."""
        bot.speaking.set()
        fake_ohbot.reset_recording()
        bot.express("happy")  # the happy pose sets both lip motors
        bot.speaking.clear()

        assert ex.TOPLIP not in fake_ohbot.motors_moved()
        assert ex.BOTTOMLIP not in fake_ohbot.motors_moved()

    def test_pose_does_drive_the_mouth_when_quiet(self, bot: Ohbot, fake_ohbot: FakeOhbot) -> None:
        fake_ohbot.reset_recording()
        bot.express("happy")
        assert ex.BOTTOMLIP in fake_ohbot.motors_moved()


class TestRecentring:
    def test_speak_returns_orientation_to_centre(self, bot: Ohbot, fake_ohbot: FakeOhbot) -> None:
        """REGRESSION: `curious` rolls the head to 8, and nothing put it back.
        The tilt persisted and compounded until the robot faced the wall."""
        fake_ohbot.reset_recording()
        bot.speak("Hello", emotion="curious")

        for motor in ex.ORIENTATION:
            final = fake_ohbot.final_position(motor)
            if final is not None:
                assert final == 5, f"motor {motor} left at {final} after speaking"

    def test_recentre_can_be_disabled(self, bot: Ohbot, fake_ohbot: FakeOhbot) -> None:
        fake_ohbot.reset_recording()
        bot.speak("Hello", emotion="curious", recentre=False)
        assert fake_ohbot.final_position(ex.HEADROLL) != 5

    def test_recentre_leaves_the_emotional_face_alone(
        self, bot: Ohbot, fake_ohbot: FakeOhbot
    ) -> None:
        """Only the aiming axes reset. Resetting the lids too would wipe the
        expression the moment the sentence ended."""
        fake_ohbot.reset_recording()
        bot.recentre()
        assert ex.LIDBLINK not in fake_ohbot.motors_moved()


class TestHeadRoll:
    def test_headroll_used_when_fitted(self, fake_ohbot: FakeOhbot) -> None:
        with Ohbot(idle=False, has_headroll=True) as bot:
            fake_ohbot.reset_recording()
            bot.express("curious")
        assert ex.HEADROLL in fake_ohbot.motors_moved()

    def test_headroll_skipped_when_absent(self, fake_ohbot: FakeOhbot) -> None:
        """Not every Ohbot has the servo fitted. Those keyframes are skipped
        rather than the whole gesture failing."""
        with Ohbot(idle=False, has_headroll=False) as bot:
            fake_ohbot.reset_recording()
            bot.express("curious")
            bot.gesture("tilt", blocking=True)
        assert ex.HEADROLL not in fake_ohbot.motors_moved()


class TestLifecycle:
    def test_close_runs_on_exit(self, fake_ohbot: FakeOhbot) -> None:
        with Ohbot(idle=False):
            pass
        assert fake_ohbot.closed

    def test_close_runs_even_when_the_body_raises(self, fake_ohbot: FakeOhbot) -> None:
        """A crash mid-conversation must still detach the motors, or they sit
        drawing current and buzzing until the robot is replugged."""
        with pytest.raises(ValueError), Ohbot(idle=False):
            raise ValueError("boom")
        assert fake_ohbot.closed


class TestSensors:
    def test_read_sensor_returns_zero_with_no_robot(self, fake_ohbot: FakeOhbot) -> None:
        """The library's readSensor calls ser.flushInput() unguarded, so it
        raises AttributeError on None when nothing is attached."""
        with Ohbot(idle=False) as bot:
            fake_ohbot.connected = False
            assert bot.read_sensor(0) == 0.0

    def test_read_sensor_reads_when_connected(self, fake_ohbot: FakeOhbot) -> None:
        with Ohbot(idle=False) as bot:
            assert bot.read_sensor(0) == 5.0


class TestExpressionApi:
    def test_unknown_emotion_lists_the_valid_ones(self, bot: Ohbot) -> None:
        with pytest.raises(KeyError, match="sympathetic"):
            bot.express("melancholic")

    def test_unknown_gesture_lists_the_valid_ones(self, bot: Ohbot) -> None:
        with pytest.raises(KeyError, match="slow_nod"):
            bot.gesture("moonwalk")

    def test_unknown_look_direction_lists_the_valid_ones(self, bot: Ohbot) -> None:
        with pytest.raises(KeyError, match="up_left"):
            bot.look_at("northwest")

    def test_empty_speech_is_ignored(self, bot: Ohbot, fake_ohbot: FakeOhbot) -> None:
        fake_ohbot.reset_recording()
        bot.speak("")
        bot.speak("   ")
        assert fake_ohbot.said == []

    def test_gesture_plays_every_keyframe(self, bot: Ohbot, fake_ohbot: FakeOhbot) -> None:
        fake_ohbot.reset_recording()
        bot.gesture("nod", blocking=True)
        assert len(fake_ohbot.moves_of(ex.HEADNOD)) == len(ex.GESTURES["nod"])

    def test_look_at_moves_eyes_and_head(self, bot: Ohbot, fake_ohbot: FakeOhbot) -> None:
        fake_ohbot.reset_recording()
        bot.look_at("up_left", blocking=True)
        moved = fake_ohbot.motors_moved()
        assert ex.EYETURN in moved and ex.EYETILT in moved
        assert ex.HEADTURN in moved

    def test_gaze_moves_eyes_only(self, bot: Ohbot, fake_ohbot: FakeOhbot) -> None:
        fake_ohbot.reset_recording()
        bot.gaze(8, 2)
        assert fake_ohbot.motors_moved() == {ex.EYETURN, ex.EYETILT}
