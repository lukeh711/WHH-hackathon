"""Microphone capture rate handling.

REGRESSION: Listener used to force sd.InputStream to SAMPLE_RATE (16 kHz)
regardless of what the device actually runs at. That's fine by luck for a
device whose driver tolerates an arbitrary requested rate, but macOS's
built-in microphone ("System default" on a MacBook) is 44.1/48 kHz native
and can't be silently retuned -- the same class of problem as the output
side (see test_audio.py's TestResampleLinear). Listener now always captures
at the device's own rate and resamples down afterwards, the way
call.CallListener already did for the call-audio case.
"""

from __future__ import annotations

import numpy as np

from ohbot_kit import voice


class TestDeviceRate:
    def test_reads_the_devices_own_rate(self, fake_sd) -> None:  # type: ignore[no-untyped-def]
        assert voice._device_rate(2) == 44100  # "MacBook Pro Microphone" in the fixture

    def test_none_means_system_default_device(self, fake_sd) -> None:  # type: ignore[no-untyped-def]
        # fake_sd.default.device = [2, 3] -> input default is index 2.
        assert voice._device_rate(None) == 44100

    def test_falls_back_to_sample_rate_on_bad_index(self, fake_sd) -> None:  # type: ignore[no-untyped-def]
        assert voice._device_rate(999) == voice.SAMPLE_RATE


class TestResampleToWhisper:
    def test_matching_rate_is_a_no_op(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        assert voice.resample_to_whisper(audio, voice.SAMPLE_RATE) is audio

    def test_downsamples_to_expected_length(self) -> None:
        audio = np.sin(np.linspace(0, 10, 48000)).astype(np.float32)
        out = voice.resample_to_whisper(audio, 48000)
        assert abs(len(out) - 16000) <= 1

    def test_empty_input_does_not_crash(self) -> None:
        assert len(voice.resample_to_whisper(np.zeros(0, dtype=np.float32), 48000)) == 0


class TestListenerCaptureRate:
    """Listener.__init__ needs faster_whisper (faked in conftest) but never
    opens a stream, so this is testable without touching InputStream."""

    def test_adopts_the_devices_native_rate(self, fake_sd) -> None:  # type: ignore[no-untyped-def]
        listener = voice.Listener(device=2)  # "MacBook Pro Microphone", 44100 Hz
        assert listener.capture_rate == 44100

    def test_capture_block_scales_with_rate(self, fake_sd) -> None:  # type: ignore[no-untyped-def]
        listener = voice.Listener(device=2)
        expected = round(voice.BLOCK * 44100 / voice.SAMPLE_RATE)
        assert listener.capture_block == expected

    def test_capture_block_is_a_positive_int(self, fake_sd) -> None:  # type: ignore[no-untyped-def]
        listener = voice.Listener(device=0)  # Plantronics input, 48000 Hz in the fixture
        assert isinstance(listener.capture_block, int)
        assert listener.capture_block > 0
