"""Tests for the kids'-content filter and fallback library.

Pure functions only -- no network, no hardware. generate() itself (the actual
Anthropic call) is exercised manually per the plan's verification checklist,
not here.
"""

from __future__ import annotations

import json
import threading

import pytest

from ohbot_kit import expression as ex
from ohbot_kit import kids_content as kc


def _content(*texts: str) -> dict:
    return {"segments": [{"text": t, "expression": "happy"} for t in texts]}


class TestFilterContent:
    def test_clean_text_passes(self) -> None:
        assert kc.filter_content(_content("A happy little bunny hops down the lane.")) == []

    def test_blocklisted_whole_word_is_reported(self) -> None:
        problems = kc.filter_content(_content("The scary monster went home."))
        assert len(problems) == 1
        assert "scary" in problems[0]
        assert "monster" in problems[0]

    def test_substring_does_not_false_positive(self) -> None:
        """REGRESSION target: "classic" must not trip on the blocked word "class"
        (which isn't even blocked) -- more importantly "scar" substrings like
        "scarf" must not trip on "scary"."""
        assert kc.filter_content(_content("She wore a soft scarf and a classic hat.")) == []

    def test_multiple_segments_checked_independently(self) -> None:
        content = _content(
            "A happy little bunny hops along.",
            "Then a scary ghost appeared!",
            "The end, everyone smiled.",
        )
        problems = kc.filter_content(content)
        assert len(problems) == 1
        assert "segment 2" in problems[0]


class TestFallbackLibrary:
    @pytest.mark.parametrize("content_type", sorted(kc.FALLBACK_LIBRARY))
    def test_each_type_has_three_to_four_items(self, content_type: str) -> None:
        items = kc.FALLBACK_LIBRARY[content_type]
        assert 3 <= len(items) <= 4

    @pytest.mark.parametrize("content_type", sorted(kc.FALLBACK_LIBRARY))
    def test_every_item_has_valid_segments(self, content_type: str) -> None:
        for item in kc.FALLBACK_LIBRARY[content_type]:
            segments = item["segments"]
            assert 3 <= len(segments) <= 5
            for seg in segments:
                assert seg["expression"] in kc.EXPRESSIONS
                assert seg["text"].strip()

    @pytest.mark.parametrize("content_type", sorted(kc.FALLBACK_LIBRARY))
    def test_get_fallback_returns_valid_item(self, content_type: str) -> None:
        item = kc.get_fallback(content_type)
        assert item in kc.FALLBACK_LIBRARY[content_type]

    def test_unknown_content_type_falls_back_to_joke(self) -> None:
        assert kc.get_fallback("not_a_real_type") in kc.FALLBACK_LIBRARY["joke"]


def test_expressions_are_a_subset_of_known_poses() -> None:
    """The app can only ever ask the robot to express() one of EXPRESSIONS, so
    every one of those tags must exist in ohbot_kit.expression.POSES -- this is
    exactly the test that would catch a forgotten `silly`/`sleepy` addition."""
    assert set(kc.EXPRESSIONS) <= set(ex.POSES)


class TestPerform:
    def test_speaks_every_segment_in_order(self, bot, fake_ohbot) -> None:
        content = _content("one", "two", "three")
        kc.perform(bot, content, threading.Event())
        assert fake_ohbot.said == ["one", "two", "three"]

    def test_gesture_is_derived_from_emotion_and_varies_by_position(
        self, bot, fake_ohbot, monkeypatch
    ) -> None:
        """Same emotion on every segment shouldn't play the identical gesture
        each time -- perform() indexes expression.gestures_for(emotion) by
        segment position, mirroring ohbot_chat.py's beats mode."""
        played: list[str] = []
        monkeypatch.setattr(bot, "gesture", lambda name, **kw: played.append(name))
        content = {
            "segments": [{"text": t, "expression": "silly"} for t in ("a", "b", "c")]
        }
        kc.perform(bot, content, threading.Event())
        suited = ex.gestures_for("silly")
        assert played == [suited[i % len(suited)] for i in range(3)]
        assert all(name in suited for name in played)

    def test_stop_event_halts_before_next_segment(self, bot, fake_ohbot) -> None:
        stop_event = threading.Event()
        content = _content("one", "two")
        stop_event.set()
        kc.perform(bot, content, stop_event)
        assert fake_ohbot.said == []

    def test_pause_before_holds_curious_face_then_speaks(
        self, bot, fake_ohbot, monkeypatch
    ) -> None:
        # Patch the `time` name inside kids_content only -- robot.py's own
        # `import time` (used by gesture keyframes' hold_seconds) is a
        # separate reference and must keep sleeping for real timing, not be
        # swept up by a blanket patch of the shared time module.
        slept: list[float] = []
        monkeypatch.setattr(kc, "time", type("FakeTime", (), {"sleep": staticmethod(slept.append)}))
        content = {
            "segments": [
                {"text": "setup", "expression": "neutral"},
                {"text": "punchline", "expression": "silly", "pause_before": True},
            ]
        }
        kc.perform(bot, content, threading.Event())
        assert slept == [kc.PUNCHLINE_PAUSE_SECONDS]
        assert fake_ohbot.said == ["setup", "punchline"]

    def test_pause_respects_stop_event(self, bot, fake_ohbot, monkeypatch) -> None:
        """A stop requested during the pause must not speak the paused segment."""
        stop_event = threading.Event()
        monkeypatch.setattr(
            kc, "time", type("FakeTime", (), {"sleep": staticmethod(lambda s: stop_event.set())})
        )
        content = {
            "segments": [
                {"text": "setup", "expression": "neutral"},
                {"text": "punchline", "expression": "silly", "pause_before": True},
            ]
        }
        kc.perform(bot, content, stop_event)
        assert fake_ohbot.said == ["setup"]


class _FakeMessages:
    def __init__(self, reply_segments: list[dict]) -> None:
        self.reply_segments = reply_segments
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        class Block:
            type = "text"
            text = json.dumps({"segments": self.reply_segments})

        class Response:
            content = [Block()]

        return Response()


class _FakeClient:
    def __init__(self, reply_segments: list[dict]) -> None:
        self.messages = _FakeMessages(reply_segments)


def _valid_segments() -> list[dict]:
    return [{"text": t, "expression": "happy"} for t in ("a", "b", "c")]


class TestGeneratePersona:
    def test_no_persona_uses_plain_system_prompt(self) -> None:
        client = _FakeClient(_valid_segments())
        kc.generate(client, "a fox", "the moon", "story")
        assert client.messages.last_kwargs["system"] == kc.SYSTEM_PROMPT

    def test_persona_system_prompt_is_appended_after_the_rules(self) -> None:
        client = _FakeClient(_valid_segments())
        persona = {"system_prompt": "You are Ohbot, a friendly pirate.", "voice": "am_michael"}
        kc.generate(client, "a fox", "the moon", "story", persona=persona)
        system = client.messages.last_kwargs["system"]
        assert system.startswith(kc.SYSTEM_PROMPT)
        assert "pirate" in system
