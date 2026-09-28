"""ohbot_chat.py's /look command: camera -> caption -> in-character reaction."""

from __future__ import annotations

import io
import threading

import ohbot_chat
from ohbot_kit import llm, vision


class _FakeEye:
    def __init__(self, description=""):
        self._description = description

    def describe(self):
        if isinstance(self._description, Exception):
            raise self._description
        return self._description


class _FakeConvo:
    """Mimics enough of llm.Conversation for look()'s history rewrite: a real
    respond_with_action appends the user turn then the assistant's reply, so
    this does too, for the same before/after len(messages) check to work."""

    def __init__(self, say="Nice hat!"):
        self.prompts: list[str] = []
        self.messages: list[dict[str, str]] = []
        self._say = say

    def respond_with_action(self, text, emotions, gestures):
        self.prompts.append(text)
        self.messages.append({"role": "user", "content": text})
        self.messages.append({"role": "assistant", "content": self._say})
        return {"say": self._say, "emotion": "happy", "gesture": "nod"}


class TestLook:
    def test_caption_reaches_the_reaction_prompt(self, bot, fake_ohbot, capsys) -> None:
        convo = _FakeConvo()
        eye = _FakeEye("A person waving.")
        ohbot_chat.look(bot, convo, eye, max_sentences=3)
        assert convo.prompts == [ohbot_chat.LOOK_PROMPT.format(description="A person waving.")]
        assert fake_ohbot.said == ["Nice hat!"]
        assert "[camera] A person waving." in capsys.readouterr().out

    def test_history_keeps_a_short_stand_in_not_the_full_instruction(self, bot, fake_ohbot) -> None:
        """A follow-up question ("do you like it?") should get answered, not
        re-joked at -- which needs the full LOOK_PROMPT instruction gone from
        history once it has done its job of steering this one reply."""
        convo = _FakeConvo()
        eye = _FakeEye("A person waving.")
        ohbot_chat.look(bot, convo, eye, max_sentences=3)
        assert convo.messages[0] == {
            "role": "user",
            "content": "[shows you a photo: A person waving.]",
        }
        assert convo.messages[1] == {"role": "assistant", "content": "Nice hat!"}

    def test_none_capture_speaks_a_fallback(self, bot, fake_ohbot) -> None:
        convo = _FakeConvo()
        eye = _FakeEye(None)
        ohbot_chat.look(bot, convo, eye, max_sentences=3)
        assert convo.prompts == []
        assert fake_ohbot.said == ["I could not see anything just then."]

    def test_empty_caption_speaks_a_fallback(self, bot, fake_ohbot) -> None:
        convo = _FakeConvo()
        eye = _FakeEye("")
        ohbot_chat.look(bot, convo, eye, max_sentences=3)
        assert convo.prompts == []
        assert fake_ohbot.said == ["I could not see anything just then."]

    def test_camera_error_speaks_a_fallback_not_a_crash(self, bot, fake_ohbot, capsys) -> None:
        convo = _FakeConvo()
        eye = _FakeEye(vision.CameraError("no camera"))
        ohbot_chat.look(bot, convo, eye, max_sentences=3)
        assert fake_ohbot.said == ["I could not see anything just then."]
        assert "[camera] no camera" in capsys.readouterr().err

    def test_ollama_error_speaks_a_fallback_not_a_crash(self, bot, fake_ohbot, capsys) -> None:
        convo = _FakeConvo()
        eye = _FakeEye(llm.OllamaError("vision model down"))
        ohbot_chat.look(bot, convo, eye, max_sentences=3)
        assert fake_ohbot.said == ["I could not see anything just then."]
        assert "[camera] vision model down" in capsys.readouterr().err


class TestWatchStdinForLook:
    """--voice's mic loop never reads stdin, so a UI's "Take a picture" button
    (which can only send text over stdin) needs this background reader to
    reach the camera while --voice is also on."""

    def test_look_lines_trigger_a_look_other_lines_are_ignored(
        self, bot, fake_ohbot, monkeypatch
    ) -> None:
        convo = _FakeConvo()
        eye = _FakeEye("A dog.")
        monkeypatch.setattr(ohbot_chat.sys, "stdin", io.StringIO("hello\n/look\n/photo\n"))
        ohbot_chat.watch_stdin_for_look(
            bot, convo, eye, max_sentences=3, mode="empathy", lock=threading.Lock()
        )
        assert convo.prompts == [ohbot_chat.LOOK_PROMPT.format(description="A dog.")] * 2
