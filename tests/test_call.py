"""Putting the robot on a video call.

Everything here is about the three ways a virtual audio cable differs from a
microphone, each of which fails silently rather than loudly:

  * it carries more channels than one, and reading channel 0 drops the rest
  * it runs at 48 kHz, not the 16 kHz Whisper is trained on
  * it is legitimately silent when nobody on the call is talking

Plus the wake-word gate, which is what stops the robot talking over people.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ohbot_kit import call
from ohbot_kit.config import Config
from ohbot_kit.voice import BLOCK, SAMPLE_RATE

AGGREGATE = "Ohbot Ears"  # the 3-channel 48 kHz fake in conftest


# -- fakes -----------------------------------------------------------------


class FakeStream:
    """An InputStream that hands out prepared blocks, then silence forever.

    Silence-after-exhaustion matters: record_utterance only returns once it has
    heard enough consecutive quiet blocks, so a stream that ran dry mid-loop
    would hang the test instead of failing it.
    """

    def __init__(self, blocks: list[Any], channels: int) -> None:
        self._blocks = list(blocks)
        self._channels = channels
        self.reads = 0

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def read(self, frames: int) -> tuple[Any, bool]:
        self.reads += 1
        if self._blocks:
            return self._blocks.pop(0), False
        return np.zeros((frames, self._channels), dtype=np.float32), False


class SpeakingBot:
    """Fake robot that reports itself speaking for the first `blocks` reads."""

    def __init__(self, blocks: int) -> None:
        self.remaining = blocks
        self.speaking = self  # so bot.speaking.is_set() lands here

    def is_set(self) -> bool:
        if self.remaining > 0:
            self.remaining -= 1
            return True
        return False


def loud(channels: int, level: float = 0.5) -> Any:
    return np.full((BLOCK, channels), level, dtype=np.float32)


def quiet(channels: int) -> Any:
    return np.zeros((BLOCK, channels), dtype=np.float32)


@pytest.fixture
def listener(fake_sd: Any) -> Any:
    """A CallListener on the fake Aggregate Device."""
    return call.CallListener(device=4, model_size="tiny.en")


@pytest.fixture
def streamed(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a FakeStream and hand back the factory used to build it."""
    made: list[FakeStream] = []

    def install(blocks: list[Any], channels: int) -> FakeStream:
        stream = FakeStream(blocks, channels)
        made.append(stream)
        monkeypatch.setattr(call.sd, "InputStream", lambda **kw: stream)
        return stream

    return install


# -- the wake-word gate ----------------------------------------------------


class TestWakeWord:
    def test_unaddressed_speech_is_rejected(self) -> None:
        """The whole point: on a call, most utterances are not for the robot."""
        assert call.wake_word("So I think we should ship on Friday.") is None
        assert call.wake_word("") is None
        assert call.wake_word(None) is None

    def test_addressed_speech_returns_just_the_request(self) -> None:
        assert call.wake_word("Ohbot, what day is it?") == "what day is it"
        assert call.wake_word("Hey Ohbot, summarise that") == "summarise that"

    def test_bare_name_is_addressed_with_an_empty_request(self) -> None:
        """Distinct from None: "Ohbot?" was aimed at the robot, so it should
        answer, but there is no question in it to answer."""
        assert call.wake_word("Ohbot?") == ""
        assert call.wake_word("hey ohbot") == ""

    def test_longest_wake_word_wins(self) -> None:
        """Matching "ohbot" first would leave "hey" glued to the request."""
        assert call.wake_word("okay Ohbot stop") == "stop"

    def test_common_mishearings_still_wake_it(self) -> None:
        """Whisper writes the name as two words often enough to matter."""
        assert call.wake_word("Oh bot, what time is it?") == "what time is it"
        assert call.wake_word("odd bot help") == "help"

    def test_custom_wake_word(self) -> None:
        words = ("computer",)
        assert call.wake_word("Computer, status?", words) == "status"
        assert call.wake_word("Ohbot, status?", words) is None


# -- channel handling ------------------------------------------------------


class TestDownmix:
    def test_single_channel_passes_through(self) -> None:
        block = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        assert call.downmix(block).tolist() == [1.0, 2.0, 3.0]

    def test_single_channel_is_copied_not_viewed(self) -> None:
        """The caller keeps these blocks in a list while the stream reuses its
        buffer, so a view would alias into whatever arrives next."""
        block = np.array([[1.0]], dtype=np.float32)
        result = call.downmix(block)
        block[0, 0] = 99.0
        assert result.tolist() == [1.0]

    def test_every_channel_contributes(self) -> None:
        """THE bug this prevents: reading channel 0 of an Aggregate Device gives
        you the call and silently discards the person sitting at the robot."""
        call_audio = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        room_only = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        assert call.downmix(call_audio).tolist() == pytest.approx([2 / 3, 2 / 3])
        # A channel-0 reader would see pure silence here and drop the utterance.
        assert call.downmix(room_only).tolist() == pytest.approx([1 / 3, 1 / 3])


class TestResolveChannels:
    def test_all_takes_every_channel_the_device_has(self) -> None:
        info = {"max_input_channels": 3}
        assert call.resolve_channels("all", info) == 3
        assert call.resolve_channels(None, info) == 3
        assert call.resolve_channels("auto", info) == 3

    def test_explicit_count_is_honoured(self) -> None:
        assert call.resolve_channels(1, {"max_input_channels": 3}) == 1

    def test_never_zero(self) -> None:
        assert call.resolve_channels(0, {}) == 1
        assert call.resolve_channels("all", {}) == 1


# -- sample rate ------------------------------------------------------------


class TestResolveRate:
    def test_device_follows_the_hardware(self) -> None:
        info = {"default_samplerate": 48000.0}
        assert call.resolve_rate("device", info) == 48000
        assert call.resolve_rate("auto", info) == 48000

    def test_explicit_rate_is_honoured(self) -> None:
        assert call.resolve_rate(16000, {"default_samplerate": 48000.0}) == 16000

    def test_falls_back_to_whisper_rate(self) -> None:
        assert call.resolve_rate("device", {}) == SAMPLE_RATE
        assert call.resolve_rate(None, {}) == SAMPLE_RATE


class TestResample:
    @pytest.mark.parametrize("rate", [16000, 32000, 44100, 48000, 96000])
    def test_length_and_dtype(self, rate: int) -> None:
        seconds = 0.5
        audio = np.zeros(int(rate * seconds), dtype=np.float32)
        out = call.resample_to_whisper(audio, rate)

        assert out.dtype == np.float32
        assert len(out) == pytest.approx(SAMPLE_RATE * seconds, rel=0.02)

    @pytest.mark.parametrize("rate", [32000, 44100, 48000, 96000])
    def test_speech_frequencies_survive(self, rate: int) -> None:
        """A 1 kHz tone has to still be a 1 kHz tone afterwards -- a resampler
        that shifted pitch would leave Whisper transcribing gibberish."""
        hz = 1000.0
        t = np.arange(int(rate * 0.5)) / rate
        out = call.resample_to_whisper(np.sin(2 * np.pi * hz * t).astype(np.float32), rate)

        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = np.fft.rfftfreq(len(out), 1 / SAMPLE_RATE)[spectrum.argmax()]
        assert peak_hz == pytest.approx(hz, abs=40)

    def test_at_whisper_rate_it_is_a_no_op(self) -> None:
        audio = np.linspace(0, 1, 100, dtype=np.float32)
        assert call.resample_to_whisper(audio, SAMPLE_RATE) is audio

    def test_content_above_the_new_nyquist_is_attenuated(self) -> None:
        """Plain decimation would fold 12 kHz down into the speech band, and
        Whisper turns that hiss into words nobody said."""
        rate = 48000
        t = np.arange(rate // 2) / rate
        hf = np.sin(2 * np.pi * 12000 * t).astype(np.float32)

        out = call.resample_to_whisper(hf, rate)
        rms_in = float(np.sqrt(np.mean(hf**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        assert rms_out < rms_in / 2

    def test_empty_input(self) -> None:
        assert len(call.resample_to_whisper(np.zeros(0, dtype=np.float32), 48000)) == 0

    def test_shorter_than_one_output_sample(self) -> None:
        assert len(call.resample_to_whisper(np.zeros(2, dtype=np.float32), 48000)) == 0


# -- the listener itself ---------------------------------------------------


class TestCallListener:
    def test_adopts_the_devices_shape(self, listener: Any) -> None:
        assert listener.channels == 3
        assert listener.capture_rate == 48000

    def test_describe_names_the_device_and_its_shape(self, listener: Any) -> None:
        text = listener.describe()
        assert AGGREGATE in text
        assert "3 channel" in text
        assert "48000" in text

    def test_explicit_settings_override_the_device(self, fake_sd: Any) -> None:
        listener = call.CallListener(device=4, channels=1, capture_rate=16000)
        assert (listener.channels, listener.capture_rate) == (1, 16000)

    def test_silence_during_calibration_is_not_an_error(self, listener: Any, streamed: Any) -> None:
        """A microphone returning digital silence means the OS denied access, so
        voice.Listener raises. A cable carrying a call nobody is speaking on is
        just quiet, and refusing to start would be wrong."""
        streamed([], listener.channels)
        listener.calibrate()
        assert listener.threshold > 0

    def test_returns_audio_at_whisper_rate(self, listener: Any, streamed: Any) -> None:
        blocks = [loud(3)] * 20 + [quiet(3)] * 60
        streamed(blocks, listener.channels)

        audio = listener.record_utterance()

        assert audio is not None
        assert audio.ndim == 1, "must be mono by the time Whisper sees it"
        # 20 loud blocks plus the silence needed to end the utterance, captured
        # at 48 kHz and handed back at 16 kHz.
        silence_limit = int(listener.silence_seconds * listener.capture_rate / BLOCK)
        expected = (20 + silence_limit) * BLOCK / 3
        assert len(audio) == pytest.approx(expected, rel=0.02)

    def test_says_so_when_the_cable_is_dead(
        self, listener: Any, streamed: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An unwired cable and a call nobody has spoken on look identical from
        in here, and the robot blinks and nods through both -- so it reads as
        working when it may be plugged into nothing. It has to say so."""
        hint_blocks = int(call.QUIET_HINT_SECONDS / (BLOCK / listener.capture_rate))
        # Silence for long enough to warrant a hint, then a real utterance so
        # record_utterance returns instead of blocking.
        streamed([quiet(3)] * (hint_blocks + 1) + [loud(3)] * 20 + [quiet(3)] * 60, 3)

        listener.record_utterance()

        out = capsys.readouterr().out
        assert "digital silence" in out
        assert AGGREGATE in out, "must name the device that is silent"
        assert "docs/TEAMS.md" in out, "must point at the fix"

    def test_quiet_but_present_audio_is_reported_differently(
        self, listener: Any, streamed: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Audio arriving too quietly is a threshold problem, not a wiring one,
        and sending someone to rebuild their routing would waste their time."""
        hint_blocks = int(call.QUIET_HINT_SECONDS / (BLOCK / listener.capture_rate))
        listener.threshold = 0.4
        faint = [loud(3, level=0.05)] * (hint_blocks + 1)
        streamed(faint + [loud(3, level=0.9)] * 20 + [quiet(3)] * 60, 3)

        listener.record_utterance()

        out = capsys.readouterr().out
        assert "below the threshold" in out
        assert "noise_multiplier" in out
        assert "digital silence" not in out

    def test_a_cough_is_not_an_utterance(self, listener: Any, streamed: Any) -> None:
        streamed([loud(3)] * 3 + [quiet(3)] * 60, listener.channels)
        assert listener.record_utterance() is None

    def test_discards_audio_while_the_robot_speaks(self, listener: Any, streamed: Any) -> None:
        """Its voice reaches its own ears -- out of the laptop speakers and back
        in via the mic half of the Aggregate Device. Without this it answers
        itself, out loud, on a live call."""
        discarded = 12
        streamed([loud(3)] * (discarded + 20) + [quiet(3)] * 60, listener.channels)

        audio = listener.record_utterance(SpeakingBot(discarded))

        assert audio is not None
        silence_limit = int(listener.silence_seconds * listener.capture_rate / BLOCK)
        kept = (20 + silence_limit) * BLOCK / 3
        assert len(audio) == pytest.approx(kept, rel=0.02)
        # The blocks heard while speaking must not be in there.
        assert len(audio) < (discarded + 20 + silence_limit) * BLOCK / 3


class TestMakeCallListener:
    def test_reads_the_call_settings_from_config(self, fake_sd: Any) -> None:
        cfg = Config(
            {
                "audio": {"input_device": AGGREGATE},
                "speech_to_text": {"model": "tiny.en", "channels": 2, "capture_rate": 44100},
            }
        )
        listener = call.make_call_listener(cfg)
        assert (listener.channels, listener.capture_rate) == (2, 44100)

    def test_defaults_suit_a_virtual_cable(self, fake_sd: Any) -> None:
        """With nothing configured it should still mix every channel and follow
        the device's rate, because that is what a cable needs."""
        cfg = Config({"audio": {"input_device": AGGREGATE}})
        listener = call.make_call_listener(cfg)
        assert (listener.channels, listener.capture_rate) == (3, 48000)
