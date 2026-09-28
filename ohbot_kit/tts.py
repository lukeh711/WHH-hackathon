"""Kokoro-82M speech for Ohbot, replacing the macOS `say` voice.

WHY THIS IS ONLY A MONKEYPATCH:

ohbot.say() looks like it does synthesis and lip sync together, so swapping the
voice ought to cost the lip movement. It doesn't -- on macOS the lip sync is
driven by loudness, not phonemes. say() opens the generated WAV, sums the sample
bytes in chunks of framerate/10, normalises to 0-10, and feeds that envelope to
_moveSpeech() as the lip position. (Real phonemes are used only on Raspberry Pi,
where synthesizer == "festival".)

So anything that writes a valid WAV to ohbotData/ohbotspeech.wav gets lip sync
for free. Replacing _generateSpeechFile is the whole integration: viseme
calculation, threading, playback and untilDone all keep working untouched.

Two constraints come from that same envelope code, and both are handled below:
  * It indexes buffer[i] + buffer[i+1]*256, so the WAV must be 16-bit PCM.
    Kokoro emits float32, so we convert.
  * It loops `range(0, length - chunk, chunk)`. Audio under ~0.2s yields an
    empty times list and say() then raises IndexError on times[-1]. Short
    replies like "Hi." are realistic, so every clip gets a silent tail.
"""

from __future__ import annotations

import os
import wave

import numpy as np
from ohbot import ohbot

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "kokoro-v1.0.onnx")
VOICES_FILE = os.path.join(MODEL_DIR, "voices-v1.0.bin")

DOWNLOAD_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
LANG = "en-us"

# Guarantees the viseme loop has at least a couple of chunks to work with.
TAIL_SILENCE_SECONDS = 0.3


class KokoroUnavailable(RuntimeError):
    """Raised when the Kokoro model files are missing."""


def available() -> bool:
    """True if both model files are present."""
    return os.path.exists(MODEL_FILE) and os.path.exists(VOICES_FILE)


def missing_files() -> list[str]:
    return [p for p in (MODEL_FILE, VOICES_FILE) if not os.path.exists(p)]


class KokoroTTS:
    """Generates Ohbot's speech WAV with Kokoro instead of macOS `say`."""

    def __init__(self, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED) -> None:
        if not available():
            raise KokoroUnavailable(
                "Missing Kokoro model files: {}\nRun: python tools/check_setup.py".format(
                    ", ".join(missing_files())
                )
            )
        # Imported here so the module can be probed for availability without
        # paying the import cost.
        from kokoro_onnx import Kokoro

        self.voice = voice
        self.speed = speed
        self._kokoro = Kokoro(MODEL_FILE, VOICES_FILE)

    def voices(self) -> list[str]:
        """Names of every voice in the voices file."""
        return sorted(self._kokoro.get_voices())

    def synth(self, text: str) -> None:
        """Render text to ohbotData/ohbotspeech.wav as 16-bit mono PCM.

        Signature matches ohbot._generateSpeechFile(text) so it can stand in
        for it directly.
        """
        samples, rate = self._kokoro.create(text, voice=self.voice, speed=self.speed, lang=LANG)

        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim > 1:  # defensive: collapse to mono
            audio = audio.mean(axis=1)

        # Scale to int16 without clipping on loud samples.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / max(peak, 1.0)
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")

        tail = np.zeros(int(rate * TAIL_SILENCE_SECONDS), dtype="<i2")
        pcm = np.concatenate([pcm, tail])

        path = ohbot.speechAudioFile
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # 16-bit, required by the viseme loop
            w.setframerate(int(rate))
            w.writeframes(pcm.tobytes())


def install(engine: KokoroTTS) -> KokoroTTS:
    """Route ohbot.say() through the given engine.

    say() looks _generateSpeechFile up as a module global at call time, so
    rebinding the attribute is enough.
    """
    ohbot._generateSpeechFile = engine.synth
    return engine


def uninstall() -> None:
    """Restore the built-in macOS `say` synthesis."""
    ohbot._generateSpeechFile = _ORIGINAL_GENERATE


_ORIGINAL_GENERATE = ohbot._generateSpeechFile


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "Hello, I am Ohbot, speaking with Kokoro."
    engine = KokoroTTS()
    print("voices available:", len(engine.voices()))
    engine.synth(text)

    with wave.open(ohbot.speechAudioFile) as w:
        print(
            f"wrote {ohbot.speechAudioFile}: {w.getframerate()} Hz, {w.getnchannels()} ch, {w.getsampwidth() * 8}-bit, {w.getnframes() / w.getframerate():.2f}s"
        )
