"""Putting Ohbot on a video call.

Teams has no bot you can install for this, and doesn't need one. Teams only
cares which audio *devices* it is pointed at, so you make Ohbot **be** the
microphone and make Teams' speaker **be** Ohbot's ears. Teams never learns a
robot joined; it just sees a mic and a speaker.

    remote participants -> Teams -> "Teams Out" (Multi-Output)
                                       |-> BlackHole 2ch --> Ohbot's ears
                                       `-> your headphones --> you

    Ohbot's voice -> "Ohbot Voice" (Multi-Output)
                        |-> BlackHole 16ch --> Teams mic --> the call
                        `-> laptop speakers --> the room

Two virtual cables, because one cable cannot carry both directions. Setting them
up is a ten-minute job in Audio MIDI Setup: see docs/TEAMS.md.

WHY THIS FILE EXISTS AT ALL

voice.Listener assumes a microphone in a room. A virtual cable differs in three
ways that each break it quietly rather than loudly:

1. CHANNEL COUNT. An Aggregate Device -- "BlackHole 2ch + built-in mic", which
   is how Ohbot hears the call *and* you -- presents three or four channels,
   with the call on some and the room on others. Reading channel 0 alone gives
   you the call and silently drops every word you say.

2. SAMPLE RATE. BlackHole runs at 48 kHz and Whisper wants 16 kHz. Asking
   CoreAudio for 16 kHz can renegotiate the device *down* instead of resampling,
   and every Multi-Output or Aggregate Device that BlackHole belongs to then
   breaks, because their members must agree on a rate. So capture at the
   device's own rate and convert in software.

3. SILENCE IS NORMAL. Listener treats digital silence during calibration as
   proof the OS denied microphone access, which is right for a real mic. On a
   quiet call it is just a quiet call, and refusing to start would be wrong.

Nothing leaves the machine: Whisper and Ollama are both local. The robot
speaking into the call is audible to everyone, though, so tell people it's there
-- some organisations treat any transcription as recording.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .voice import (
    BLOCK,
    CALIBRATION_SECONDS,
    FLOOR,
    MAX_UTTERANCE_SECONDS,
    SAMPLE_RATE,
    Listener,
    _rms,
    resample_to_whisper,
)

# Whisper hears the robot's name several ways -- voice.NAME_HINT biases the
# decoder towards "Ohbot", but "oh bot" and "odd bot" still come out regularly,
# so normalise them rather than losing the utterance.
_MISHEARD = (
    (re.compile(r"\boh\s+bot\b"), "ohbot"),
    (re.compile(r"\bodd\s+bot\b"), "ohbot"),
    (re.compile(r"\bawe?\s+bot\b"), "ohbot"),
)
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")

WAKE_WORDS = ("hey ohbot", "hi ohbot", "okay ohbot", "ok ohbot", "ohbot")

# How long to wait in silence before saying so. An unwired cable and a call
# nobody has spoken on yet are indistinguishable from inside the loop, and the
# robot blinks and nods through both, so silence has to be reported rather than
# left to look like patience.
QUIET_HINT_SECONDS = 12.0


def wake_word(text: str | None, words: Iterable[str] = WAKE_WORDS) -> str | None:
    """Split an utterance into (was I addressed?, what was asked).

    Returns None when the robot was not addressed, and the remaining request
    when it was -- which is the empty string for a bare "Ohbot?".

    That three-way answer is the point. On a real call a robot that replies to
    every utterance talks over people constantly, so the loop needs to tell
    "not for me" apart from "for me, and here is the question".
    """
    cleaned = _PUNCT.sub(" ", (text or "").lower())
    for pattern, replacement in _MISHEARD:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = _SPACES.sub(" ", cleaned).strip()

    # Longest first, so "hey ohbot" is not matched as a bare "ohbot".
    for word in sorted(words, key=len, reverse=True):
        if cleaned.startswith(word):
            return cleaned[len(word) :].strip()
    return None


# -- signal shaping ---------------------------------------------------------
#
# Free functions rather than methods so they can be tested without loading a
# 150 MB speech model to get at them.


def downmix(block: NDArray[Any]) -> NDArray[Any]:
    """Collapse however many channels the device gave us into one signal."""
    if block.shape[1] == 1:
        return block[:, 0].copy()
    return block.mean(axis=1)


def resolve_channels(channels: int | str | None, info: Mapping[str, Any]) -> int:
    """How many channels to read: an explicit count, or all the device has."""
    # `is None` is spelled out rather than folded into the tuple below because
    # mypy 1.19 (the pinned version, and what CI runs) does not narrow None out
    # of `channels in (None, ...)`, so the int() call reads as int(None) and
    # fails the type check. Newer mypy narrows it and passes either way.
    if channels is None or channels in ("all", "auto"):
        return max(1, int(info.get("max_input_channels", 1)))
    return max(1, int(channels))


def resolve_rate(capture_rate: int | str | None, info: Mapping[str, Any]) -> int:
    """What rate to capture at: an explicit rate, or the device's own."""
    if capture_rate in ("device", "auto"):
        rate = info.get("default_samplerate")
        return int(rate) if rate else SAMPLE_RATE
    return int(capture_rate) if capture_rate else SAMPLE_RATE


class CallListener(Listener):
    """A Listener for audio arriving down a virtual cable, not from a room.

        listener = CallListener(device=idx, channels="all", capture_rate="device")

    channels: an integer, or "all" to mix down every channel the device offers
    -- which is what you want for an Aggregate Device, and the only way Ohbot
    hears both the call and you.

    capture_rate: a rate in Hz, or "device" to capture at whatever the device
    runs at and resample to 16 kHz in software.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        channels = kwargs.pop("channels", "all")
        capture_rate = kwargs.pop("capture_rate", "device")
        super().__init__(*args, **kwargs)

        info = self._device_info()
        self.channels = resolve_channels(channels, info)
        self.capture_rate = resolve_rate(capture_rate, info)

    # -- device interrogation ----------------------------------------------

    def _device_info(self) -> Mapping[str, Any]:
        try:
            index = self.device if self.device is not None else sd.default.device[0]
            return sd.query_devices(index)
        except Exception:
            return {}

    def describe(self) -> str:
        name = self._device_info().get("name", "system default")
        return f"{name} - {self.channels} channel(s) at {self.capture_rate} Hz"

    # -- capture -----------------------------------------------------------

    def _stream(self) -> Any:
        return sd.InputStream(
            samplerate=self.capture_rate,
            channels=self.channels,
            blocksize=BLOCK,
            device=self.device,
        )

    def _mono(self, block: NDArray[Any]) -> NDArray[Any]:
        return downmix(block)

    def _to_whisper_rate(self, audio: NDArray[Any]) -> NDArray[Any]:
        return resample_to_whisper(audio, self.capture_rate)

    def _report_quiet(self, loudest: float) -> None:
        """Explain a long silence, telling a dead cable from a quiet call."""
        name = self._device_info().get("name", "the input device")
        if loudest <= FLOOR:
            print(
                f"[call] {QUIET_HINT_SECONDS}s of digital silence on {name} "
                f"(loudest {loudest:.5f}).\n"
                "       Nothing is feeding it. On a call, the meeting app's SPEAKER must be a\n"
                "       Multi-Output containing this device -- see docs/TEAMS.md. To test with\n"
                "       no call at all, set the system output to it and play a video.",
                flush=True,
            )
        else:
            print(
                f"[call] audio is arriving on {name} but stays below the threshold "
                f"(loudest {loudest:.5f}, threshold {self.threshold:.5f}). Turn the source "
                "up, or lower speech_to_text.noise_multiplier.",
                flush=True,
            )

    def calibrate(self) -> None:
        """Measure the noise floor of the cable.

        Unlike the microphone version this does NOT treat pure silence as a
        failure. A cable carrying a call nobody is speaking on is silent, and
        that is the normal state before the meeting starts. Use
        tools/check_call_audio.py to prove the cable is actually wired.
        """
        print(f"Calibrating: {self.describe()}", flush=True)
        levels = []
        with self._stream() as stream:
            deadline_blocks = int(CALIBRATION_SECONDS * self.capture_rate / BLOCK)
            for _ in range(max(1, deadline_blocks)):
                block, _overflow = stream.read(BLOCK)
                levels.append(_rms(self._mono(block)))

        baseline = float(np.median(levels)) if levels else 0.0
        self.threshold = max(baseline * self.noise_multiplier, FLOOR)

        if baseline == 0.0:
            print(
                "Cable is silent right now, which is fine if the call is quiet.\n"
                "If nobody is ever heard, the cable is not wired: run\n"
                "    python tools/check_call_audio.py"
            )
        print(f"Noise floor {baseline:.5f}, threshold {self.threshold:.5f}")

    def record_utterance(self, bot: Any = None) -> NDArray[Any] | None:
        """Block until someone on the call speaks, then return their audio.

        Same shape as Listener.record_utterance -- float32 array at 16 kHz, or
        None if what arrived was too short to be speech.
        """
        frames: list[NDArray[Any]] = []
        speech_frames = 0
        silence_frames = 0
        started = False

        block_seconds = BLOCK / float(self.capture_rate)
        silence_limit = max(1, int(self.silence_seconds / block_seconds))
        max_frames = max(1, int(MAX_UTTERANCE_SECONDS / block_seconds))

        # A cable that is not wired up looks exactly like a call where nobody
        # has spoken yet: this loop just blocks. Meanwhile the robot blinks and
        # backchannel-nods, so it reads as "working, listening" when it may in
        # fact be plugged into nothing. Say so instead of waiting in silence.
        hint_blocks = max(1, int(QUIET_HINT_SECONDS / block_seconds))
        waited = 0
        loudest = 0.0

        with self._stream() as stream:
            while True:
                block, overflowed = stream.read(BLOCK)
                if overflowed:
                    continue
                mono = self._mono(block)

                # Ohbot's own voice goes out over the laptop speakers, and if
                # its ears include the built-in mic it hears itself. Drop audio
                # while it speaks or it replies to itself on a live call.
                if bot is not None and bot.speaking.is_set():
                    frames, started, speech_frames, silence_frames = [], False, 0, 0
                    continue

                level = _rms(mono)
                loud = level > self.threshold

                if not started:
                    if loud:
                        started = True
                        frames.append(mono)
                        speech_frames = 1
                        waited = 0
                        loudest = 0.0
                        continue

                    loudest = max(loudest, level)
                    waited += 1
                    if waited >= hint_blocks:
                        self._report_quiet(loudest)
                        waited = 0
                        loudest = 0.0
                    continue

                frames.append(mono)
                if loud:
                    speech_frames += 1
                    silence_frames = 0
                else:
                    silence_frames += 1

                if silence_frames >= silence_limit or len(frames) >= max_frames:
                    break

        if speech_frames * block_seconds < self.min_speech_seconds:
            return None
        return self._to_whisper_rate(np.concatenate(frames))


def make_call_listener(cfg: Any, device: int | None = None) -> CallListener:
    """Build a CallListener from config, mirroring ohbot_kit.make_listener.

    Reads the same speech_to_text section, plus two keys that only matter on a
    call: speech_to_text.channels and speech_to_text.capture_rate.
    """
    from . import audio, voice

    if device is None:
        device = audio.resolve(cfg.get("audio.input_device"), audio.INPUT)

    return CallListener(
        device=device,
        model_size=cfg.get("speech_to_text.model", voice.MODEL_SIZE),
        silence_seconds=cfg.get("speech_to_text.silence_seconds", voice.SILENCE_SECONDS),
        min_speech_seconds=cfg.get("speech_to_text.min_speech_seconds", voice.MIN_SPEECH_SECONDS),
        noise_multiplier=cfg.get("speech_to_text.noise_multiplier", voice.NOISE_MULTIPLIER),
        channels=cfg.get("speech_to_text.channels", "all"),
        capture_rate=cfg.get("speech_to_text.capture_rate", "device"),
    )


if __name__ == "__main__":
    # Quick check of the wake-word gate without loading a speech model.
    for line in (
        "Ohbot, what do you think?",
        "Hey Ohbot",
        "oh bot how long until the deadline",
        "So anyway, I sent the deck yesterday.",
    ):
        print(f"{line!r:<45} -> {wake_word(line)!r}")
