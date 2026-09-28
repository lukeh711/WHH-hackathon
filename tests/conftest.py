"""Test fixtures, and the fakes that make the suite runnable without hardware.

WHY FAKES ARE INSTALLED BEFORE ANY IMPORT

The real `ohbot` package calls init() unconditionally at import time. On Linux
that shells out to `aplay`, enumerates serial ports and writes an ohbotData/
folder; on macOS it plays a warm-up sound. Five of the eight ohbot_kit modules
import it at module level, so `import ohbot_kit` alone triggers all of that.
`sounddevice` is worse: importing it requires PortAudio to be installed on the
system, which CI runners do not have by default.

So both are replaced in sys.modules here, before ohbot_kit is ever imported.
That makes the suite hermetic, fast, and identical on Ubuntu, macOS and Windows
with no robot, no audio device and no apt packages.

The fake ohbot is not just a silencer: it RECORDS every motor command. That
recording is the assertion surface for the behaviour that has actually broken
in this project -- the mouth being driven while lip sync owns it, the head
failing to recentre after an utterance, and head-roll keyframes running on a
robot with no head-roll servo.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

# --------------------------------------------------------------------------
# Fake ohbot
# --------------------------------------------------------------------------


class FakeOhbot(types.ModuleType):
    """Stand-in for the ohbot library that records what it was told to do."""

    # Motor numbers, matching the real library.
    HEADNOD = 0
    HEADTURN = 1
    EYETURN = 2
    LIDBLINK = 3
    TOPLIP = 4
    BOTTOMLIP = 5
    EYETILT = 6
    HEADROLL = 7

    def __init__(self, name: str = "ohbot") -> None:
        super().__init__(name)
        self.reset_recording()
        self.connected = True
        # None mirrors the real module before init() opens the port: nothing
        # for release/__exit__'s "close the actual serial connection" step
        # to close, so that guard is exercised as a no-op here.
        self.ser = None
        self.speechAudioFile = "ohbotData/ohbotspeech.wav"
        self.motorPos = [5] * 8
        self.motorMins = [0] * 8
        self.motorMaxs = [180] * 8
        self.motorType: list[str | None] = [None] * 8
        # Speech duration is simulated rather than real: tests assert on
        # ordering and on which motors moved, never on audio.
        self.say_duration = 0.0

    def reset_recording(self) -> None:
        self.moves: list[tuple[int, float, float]] = []
        self.said: list[str] = []
        self.eye_colours: list[tuple[int, int, int]] = []
        self.closed = False
        self.reset_calls = 0

    # -- the API ohbot_kit uses --------------------------------------------

    def move(self, m: int, pos: float, spd: float = 5, eye: int = 0) -> None:
        self.moves.append((m, pos, spd))
        self.motorPos[m] = pos

    def say(
        self,
        text: str,
        untilDone: bool = True,
        lipSync: bool = True,
        hdmiAudio: bool = False,
        soundDelay: float = 0,
    ) -> None:
        self.said.append(text)
        # Emulate lip sync so tests can tell gesture moves from mouth moves.
        self.moves.append((self.BOTTOMLIP, 7, 10))
        self.moves.append((self.TOPLIP, 6, 10))

    def setEyeColour(self, r: int, g: int, b: int, swapRandG: bool = False) -> None:
        self.eye_colours.append((r, g, b))

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True

    def readSensor(self, index: int) -> float:
        return 5.0

    def _serwrite(self, s: str) -> None:
        pass

    def _generateSpeechFile(self, text: str) -> None:
        pass

    def _playSpeech(self, addSilence: bool) -> None:
        pass

    def init(self, portName: str | None = None) -> bool:
        return True

    # Convenience for assertions -------------------------------------------

    def motors_moved(self) -> set[int]:
        return {m for m, _, _ in self.moves}

    def moves_of(self, motor: int) -> list[tuple[int, float, float]]:
        return [mv for mv in self.moves if mv[0] == motor]

    def final_position(self, motor: int) -> float | None:
        hits = self.moves_of(motor)
        return hits[-1][1] if hits else None


class FakeSoundDevice(types.ModuleType):
    """Minimal sounddevice: importable without PortAudio installed."""

    class PortAudioError(Exception):
        pass

    def __init__(self, name: str = "sounddevice") -> None:
        super().__init__(name)
        self.played: list[Any] = []
        # A headset deliberately appears TWICE under one name -- once as an
        # input, once as an output. That collision is the reason device
        # resolution has to be direction-aware, so the fixture reproduces it.
        self._devices = [
            {
                "name": "Plantronics Blackwire 3220",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "Plantronics Blackwire 3220",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
            {
                "name": "MacBook Pro Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 44100.0,
            },
            {
                "name": "MacBook Pro Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
            # An Aggregate Device, as used to put the robot on a video call:
            # "BlackHole 2ch + built-in mic" carries the call on one pair of
            # channels and the room on another, at 48 kHz rather than the
            # 16 kHz Whisper wants. Both of those trip up code written for a
            # plain microphone, so the fixture provides one to test against.
            {
                "name": "Ohbot Ears (Aggregate)",
                "max_input_channels": 3,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
        ]

        class _Default:
            device = [2, 3]

        self.default = _Default()

    def query_devices(self, index: int | None = None) -> Any:
        if index is None:
            return self._devices
        return self._devices[index]

    def play(self, data: Any, rate: int, device: int | None = None) -> None:
        self.played.append((rate, device))

    def wait(self) -> None:
        pass

    def InputStream(self, **kwargs: Any) -> Any:  # noqa: N802 - mirrors the real API
        raise NotImplementedError("microphone capture is not exercised in unit tests")


class FakeFasterWhisper(types.ModuleType):
    """Stand-in for faster_whisper, which CI does not install.

    ohbot_kit.voice imports WhisperModel at module level, and ohbot_kit.call
    subclasses its Listener, so importing the call module would otherwise pull
    in ctranslate2 and a 150 MB model download. Constructing the fake model is
    free, so tests can build a CallListener and exercise its real logic.
    """

    def __init__(self, name: str = "faster_whisper") -> None:
        super().__init__(name)

        class WhisperModel:
            def __init__(self, size: str, device: str = "cpu", compute_type: str = "int8"):
                self.size = size
                self.transcribed: list[Any] = []

            def transcribe(self, audio: Any, **kwargs: Any) -> Any:
                self.transcribed.append(audio)
                return [], None

        self.WhisperModel = WhisperModel


class FakeCv2(types.ModuleType):
    """Stand-in for cv2, which CI does not install.

    ohbot_kit.vision imports cv2 lazily inside Camera.__init__, so this only
    needs to cover what that module calls: VideoCapture (open/read/release)
    plus resize and imencode. Scriptable via the flags below, set before
    constructing a vision.Camera.
    """

    def __init__(self, name: str = "cv2") -> None:
        super().__init__(name)
        self.open_succeeds = True
        self.frame_present = True
        self.imencode_succeeds = True
        self.opened_indexes: list[int] = []
        self.released: list[int] = []
        self.resized: list[Any] = []

        module = self

        class VideoCapture:
            def __init__(self, index: int) -> None:
                self.index = index
                module.opened_indexes.append(index)
                self._open = module.open_succeeds
                self.read_calls = 0

            def isOpened(self) -> bool:  # noqa: N802 - mirrors the real API
                return self._open

            def read(self) -> Any:
                self.read_calls += 1
                if not module.frame_present:
                    return False, None
                return True, f"frame-{self.read_calls}"

            def release(self) -> None:
                self._open = False
                module.released.append(self.index)

        self.VideoCapture = VideoCapture

    def resize(self, frame: Any, size: Any) -> Any:
        self.resized.append((frame, size))
        return frame

    def imencode(self, ext: str, frame: Any) -> Any:
        if not self.imencode_succeeds:
            return False, None

        class _Buf:
            def tobytes(self) -> bytes:
                return b"jpeg-bytes"

        return True, _Buf()


# Install the fakes before ohbot_kit is imported anywhere.
_fake_ohbot = FakeOhbot()
_fake_sd = FakeSoundDevice()
_fake_whisper = FakeFasterWhisper()
_fake_cv2 = FakeCv2()

# The real package is `ohbot.ohbot`, imported as `from ohbot import ohbot`.
_ohbot_pkg = types.ModuleType("ohbot")
_ohbot_pkg.ohbot = _fake_ohbot  # type: ignore[attr-defined]
sys.modules.setdefault("ohbot", _ohbot_pkg)
sys.modules.setdefault("ohbot.ohbot", _fake_ohbot)
sys.modules.setdefault("sounddevice", _fake_sd)
sys.modules.setdefault("faster_whisper", _fake_whisper)
sys.modules.setdefault("cv2", _fake_cv2)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def fake_ohbot() -> FakeOhbot:
    """The recording fake, cleared before each test."""
    _fake_ohbot.reset_recording()
    _fake_ohbot.connected = True
    return _fake_ohbot


@pytest.fixture
def fake_sd() -> FakeSoundDevice:
    return _fake_sd


@pytest.fixture
def fake_cv2() -> FakeCv2:
    """The scriptable fake, reset to its defaults before each test."""
    _fake_cv2.open_succeeds = True
    _fake_cv2.frame_present = True
    _fake_cv2.imencode_succeeds = True
    _fake_cv2.opened_indexes = []
    _fake_cv2.released = []
    _fake_cv2.resized = []
    return _fake_cv2


@pytest.fixture
def bot(fake_ohbot: FakeOhbot):  # type: ignore[no-untyped-def]
    """An Ohbot wrapper with idle motion off, wrapped around the fake.

    Idle motion is disabled because its random blinking would make assertions
    about which motors moved non-deterministic.
    """
    from ohbot_kit.robot import Ohbot

    with Ohbot(idle=False) as robot:
        fake_ohbot.reset_recording()
        yield robot
