"""Kokoro WAV output: the format constraints ohbot.say() imposes."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from ohbot_kit import tts


class FakeKokoro:
    """Stands in for the ONNX model: returns silence of a chosen length."""

    def __init__(self, seconds: float = 1.0, rate: int = 24000) -> None:
        self.seconds = seconds
        self.rate = rate

    def create(self, text, voice, speed, lang):  # type: ignore[no-untyped-def]
        samples = np.linspace(-0.5, 0.5, int(self.rate * self.seconds), dtype=np.float32)
        return samples, self.rate

    def get_voices(self):  # type: ignore[no-untyped-def]
        return ["af_heart", "bf_emma"]


@pytest.fixture
def engine(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ohbotData").mkdir()

    eng = tts.KokoroTTS.__new__(tts.KokoroTTS)  # skip the model file check
    eng.voice = "af_heart"
    eng.speed = 1.0
    eng._kokoro = FakeKokoro()
    return eng


def read_wav(path: str):  # type: ignore[no-untyped-def]
    with wave.open(path, "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()


class TestWavFormat:
    def test_is_16_bit_mono(self, engine, fake_ohbot) -> None:  # type: ignore[no-untyped-def]
        """ohbot's viseme loop indexes buffer[i] + buffer[i+1]*256, so the WAV
        must be 16-bit. Kokoro emits float32, hence the conversion."""
        engine.synth("hello")
        channels, width, _rate, _frames = read_wav(fake_ohbot.speechAudioFile)
        assert channels == 1
        assert width == 2

    def test_short_clips_are_padded(self, engine, fake_ohbot) -> None:  # type: ignore[no-untyped-def]
        """REGRESSION: say() loops range(0, length - chunk, chunk) with
        chunk = framerate/10. Audio under ~0.2s leaves `times` empty and say()
        then raises IndexError on times[-1]. A one-word reply crashed the demo."""
        engine._kokoro = FakeKokoro(seconds=0.02)  # 20ms, far below one chunk
        engine.synth("Hi.")

        _ch, _w, rate, frames = read_wav(fake_ohbot.speechAudioFile)
        chunk = rate / 10
        assert frames > 2 * chunk, "too short: say() will raise IndexError on times[-1]"

    def test_audio_is_not_clipped(self, engine, fake_ohbot) -> None:  # type: ignore[no-untyped-def]
        engine.synth("hello")
        with wave.open(fake_ohbot.speechAudioFile, "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        assert pcm.max() <= 32767 and pcm.min() >= -32768


class TestAvailability:
    def test_missing_model_files_name_the_fix(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.chdir(tmp_path)
        with pytest.raises(tts.KokoroUnavailable, match="check_setup"):
            tts.KokoroTTS()

    def test_available_reports_missing(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.chdir(tmp_path)
        assert not tts.available()
        assert len(tts.missing_files()) == 2


class TestInstall:
    def test_install_routes_say_through_kokoro(self, engine, fake_ohbot) -> None:  # type: ignore[no-untyped-def]
        """say() looks _generateSpeechFile up as a module global at call time,
        so rebinding the attribute is enough to swap the whole voice."""
        original = fake_ohbot._generateSpeechFile
        try:
            tts.install(engine)
            assert fake_ohbot._generateSpeechFile == engine.synth
        finally:
            fake_ohbot._generateSpeechFile = original
