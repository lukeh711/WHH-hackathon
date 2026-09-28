"""Audio device selection for Ohbot.

Two jobs: resolve a device name to a PortAudio index, and route the robot's
speech to a chosen output device.

WHY OUTPUT NEEDS A MONKEYPATCH:

ohbot._playSpeech() calls playsound(speechFile) from playsound3, which always
uses the system default output -- there is no device argument. So choosing an
output device means replacing _playSpeech, the same way tts.py replaces
_generateSpeechFile. This is safe because say() launches it as
threading.Thread(target=_playSpeech, ...), resolving the module global at call
time, so rebinding the attribute takes effect.

Input needs none of this: voice.Listener already passes a device through to
sd.InputStream. It has simply never been set.

WHY MATCHING IS DIRECTION-AWARE:

A headset appears TWICE in the device list under an identical name -- once with
input channels, once with output channels. Matching on name alone picks the
wrong one half the time, so candidates are filtered by direction first.

Names are used rather than indices because indices shift when devices are
plugged or unplugged, leaving a saved index silently pointing at the wrong
hardware.
"""

from __future__ import annotations

import wave
from typing import Any

import numpy as np
import sounddevice as sd
from ohbot import ohbot

INPUT = "input"
OUTPUT = "output"


class DeviceNotFound(RuntimeError):
    """Raised when a configured device name matches nothing."""


def devices(kind: str) -> list[tuple[int, Any]]:
    """(index, info) pairs for devices usable in the given direction."""
    key = "max_input_channels" if kind == INPUT else "max_output_channels"
    return [(i, d) for i, d in enumerate(sd.query_devices()) if d[key] > 0]


def describe() -> str:
    """Human-readable listing of inputs and outputs, for --list-devices."""
    lines = []
    for kind in (INPUT, OUTPUT):
        lines.append(f"{kind.capitalize()}s:")
        default = sd.default.device[0 if kind == INPUT else 1]
        for i, d in devices(kind):
            mark = "  <- system default" if i == default else ""
            lines.append("  [{}] {}{}".format(i, d["name"], mark))
    return "\n".join(lines)


def resolve(name: str | None, kind: str) -> int | None:
    """Map a device name substring to an index.

    None or empty means "use the system default", signalled by returning None.
    """
    if not name:
        return None

    candidates = devices(kind)
    needle = str(name).lower()

    for i, d in candidates:
        if needle in d["name"].lower():
            return i

    raise DeviceNotFound(f"No {kind} device matching {name!r}.\n{describe()}")


def device_name(index: int | None) -> str:
    """Name for an index, or 'system default' for None."""
    if index is None:
        return "system default"
    return sd.query_devices(index)["name"]


_ORIGINAL_PLAY = ohbot._playSpeech


def _fade_edges(pcm: np.ndarray, rate: int, fade_ms: float = 8.0) -> np.ndarray:
    """Ramp the first/last few ms of `pcm` to and from silence.

    sd.play() starts and stops mid-waveform, and that discontinuity is an
    audible click -- small speaker enclosures (e.g. a MacBook's built-in
    ones) ring it into a crackle. A fade this short removes the click
    without being audible as a fade itself.
    """
    n = min(int(rate * fade_ms / 1000), len(pcm) // 2)
    if n <= 0:
        return pcm
    ramp = np.linspace(0.0, 1.0, n)
    if pcm.ndim > 1:
        ramp = ramp[:, None]
    faded = pcm.astype(np.float64)
    faded[:n] *= ramp
    faded[-n:] *= ramp[::-1]
    return faded.astype(pcm.dtype)


def _resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linearly resample `pcm` from src_rate to dst_rate.

    Kokoro renders at its own native rate (24kHz), which doesn't match this
    Mac's output devices (all 48kHz). Asking sd.play() to open a stream at
    the WAV's rate forces CoreAudio to retune the device's *nominal* sample
    rate on every single utterance -- audible on built-in speakers as a
    "tock" transient, and the source of the PaMacCore "err=-50" in the
    console. Resampling to the device's own rate means playback never asks
    for a rate switch. Linear interpolation is a soft/cheap resampler, but
    it's more than good enough for spoken-word audio and needs no extra
    dependency (no scipy in this project).
    """
    if src_rate == dst_rate or len(pcm) == 0:
        return pcm
    duration = len(pcm) / src_rate
    n_dst = max(int(round(duration * dst_rate)), 1)
    src_times = np.arange(len(pcm)) / src_rate
    dst_times = np.arange(n_dst) / dst_rate
    if pcm.ndim == 1:
        resampled = np.interp(dst_times, src_times, pcm.astype(np.float64))
    else:
        resampled = np.stack(
            [np.interp(dst_times, src_times, pcm[:, ch].astype(np.float64)) for ch in range(pcm.shape[1])],
            axis=1,
        )
    return resampled.astype(pcm.dtype)


def install_output(device: int | None) -> None:
    """Route ohbot speech playback to a specific output device.

    Passing None restores the default playsound path.
    """
    if device is None:
        ohbot._playSpeech = _ORIGINAL_PLAY
        return

    def _play(addSilence: bool) -> None:
        try:
            with wave.open(ohbot.speechAudioFile, "rb") as w:
                rate = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
                channels = w.getnchannels()
            if channels > 1:
                pcm = pcm.reshape(-1, channels)

            device_rate = int(sd.query_devices(device)["default_samplerate"])
            if device_rate != rate:
                pcm = _resample_linear(pcm, rate, device_rate)
                rate = device_rate

            pcm = _fade_edges(pcm, rate)
        except Exception as e:
            # Nothing has been played yet, so falling back is safe here.
            print(f"[audio] could not prepare {ohbot.speechAudioFile} ({e}), using system default")
            _ORIGINAL_PLAY(addSilence)
            return

        try:
            # blocksize is deliberately large: sd.play()'s audio callback runs
            # in Python and needs the GIL on every call PortAudio makes to it.
            # Inside Streamlit (Kids' Content), that callback competes for the
            # GIL with the script-runner/session/websocket threads Streamlit
            # itself keeps busy -- contention a bare subprocess like
            # ohbot_chat.py never has. A small blocksize means the callback
            # has to win that race constantly during playback; a big one
            # means it's called far less often, so an occasional lost race
            # doesn't starve the buffer and crackle.
            sd.play(pcm, rate, device=device, latency="high", blocksize=8192)
            sd.wait()
        except Exception as e:
            # By the time sd.play()/sd.wait() can fail, some audio may
            # already be out of the speaker -- falling back to _ORIGINAL_PLAY
            # here would replay the whole clip from the start on top of that,
            # heard as an echo. Log and stop instead of doubling up.
            print(f"[audio] playback on device {device} failed partway through ({e})")

    ohbot._playSpeech = _play


if __name__ == "__main__":
    import sys

    print(describe())
    if len(sys.argv) > 1:
        name = sys.argv[1]
        print()
        for kind in (INPUT, OUTPUT):
            try:
                idx = resolve(name, kind)
                print(f"{name!r} as {kind}: [{idx}] {device_name(idx)}")
            except DeviceNotFound as e:
                print(f"{name!r} as {kind}: {e.args[0].splitlines()[0]}")
