"""Audio device resolution."""

from __future__ import annotations

import numpy as np
import pytest

from ohbot_kit import audio


class TestResolve:
    def test_direction_aware_matching(self) -> None:
        """THE case this exists for: a headset appears twice under one name,
        once as an input and once as an output. Matching on name alone picks
        the wrong one half the time."""
        mic = audio.resolve("Plantronics", audio.INPUT)
        speaker = audio.resolve("Plantronics", audio.OUTPUT)

        assert mic != speaker, "same index returned for both directions"

        input_indexes = [i for i, _ in audio.devices(audio.INPUT)]
        output_indexes = [i for i, _ in audio.devices(audio.OUTPUT)]
        assert mic in input_indexes
        assert speaker in output_indexes

    def test_case_insensitive_substring(self) -> None:
        assert audio.resolve("plantronics", audio.INPUT) == audio.resolve(
            "Plantronics Blackwire", audio.INPUT
        )

    def test_none_means_system_default(self) -> None:
        assert audio.resolve(None, audio.INPUT) is None
        assert audio.resolve("", audio.OUTPUT) is None

    def test_unknown_name_lists_real_devices(self) -> None:
        """The error has to be actionable -- a participant hits this when
        their headset is named differently from a teammate's."""
        with pytest.raises(audio.DeviceNotFound, match="MacBook Pro"):
            audio.resolve("Nonexistent Device", audio.OUTPUT)


class TestDescribe:
    def test_lists_inputs_and_outputs_separately(self) -> None:
        text = audio.describe()
        assert "Inputs:" in text and "Outputs:" in text

    def test_device_name_handles_default(self) -> None:
        assert audio.device_name(None) == "system default"


class TestFadeEdges:
    """REGRESSION: sd.play() starting/stopping mid-waveform is an audible
    click that small speaker enclosures ring into a crackle -- _fade_edges
    removes the discontinuity at the start and end of playback."""

    def test_starts_and_ends_at_silence(self) -> None:
        pcm = np.full(2000, 30000, dtype="<i2")
        faded = audio._fade_edges(pcm, rate=48000)
        assert faded[0] == 0
        assert faded[-1] == 0
        assert faded.dtype == pcm.dtype

    def test_middle_is_unchanged(self) -> None:
        pcm = np.full(2000, 30000, dtype="<i2")
        faded = audio._fade_edges(pcm, rate=48000)
        assert faded[len(faded) // 2] == 30000

    def test_multichannel_shape_preserved(self) -> None:
        pcm = np.full((2000, 2), 30000, dtype="<i2")
        faded = audio._fade_edges(pcm, rate=48000)
        assert faded.shape == pcm.shape
        assert tuple(faded[0]) == (0, 0)

    def test_short_buffer_does_not_crash(self) -> None:
        """A one-word utterance can be shorter than the fade window."""
        pcm = np.full(10, 30000, dtype="<i2")
        faded = audio._fade_edges(pcm, rate=48000)
        assert len(faded) == len(pcm)


class TestResampleLinear:
    """REGRESSION: Kokoro renders at 24kHz but this Mac's output devices are
    all 48kHz-native. Forcing sd.play() to the WAV's own rate made CoreAudio
    retune the device's nominal sample rate on every utterance -- audible on
    built-in speakers as a "tock", and the source of a PaMacCore err=-50.
    Resampling to the device's rate avoids the retune entirely."""

    def test_matching_rate_is_a_no_op(self) -> None:
        pcm = np.arange(100, dtype="<i2")
        assert audio._resample_linear(pcm, 48000, 48000) is pcm

    def test_upsamples_to_target_length(self) -> None:
        pcm = np.linspace(0, 30000, 240, dtype="<i2")
        resampled = audio._resample_linear(pcm, 24000, 48000)
        assert resampled.dtype == pcm.dtype
        assert abs(len(resampled) - 480) <= 1

    def test_downsamples_to_target_length(self) -> None:
        pcm = np.linspace(0, 30000, 480, dtype="<i2")
        resampled = audio._resample_linear(pcm, 48000, 24000)
        assert abs(len(resampled) - 240) <= 1

    def test_preserves_endpoint_values(self) -> None:
        pcm = np.array([0, 10000, 20000, 30000], dtype="<i2")
        resampled = audio._resample_linear(pcm, 24000, 48000)
        assert resampled[0] == 0
        assert resampled[-1] == pytest.approx(30000, abs=1)

    def test_multichannel_shape_preserved(self) -> None:
        pcm = np.column_stack([np.arange(240, dtype="<i2"), np.arange(240, dtype="<i2")])
        resampled = audio._resample_linear(pcm, 24000, 48000)
        assert resampled.ndim == 2
        assert resampled.shape[1] == 2

    def test_empty_buffer_does_not_crash(self) -> None:
        pcm = np.array([], dtype="<i2")
        assert len(audio._resample_linear(pcm, 24000, 48000)) == 0
