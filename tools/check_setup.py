"""Run this before the demo.

Downloads what's missing, warms up every model so the first live response isn't
slow, and checks each thing that has to be true for the demo to work -- so you
find out here rather than in front of an audience.

    python tools/check_setup.py            # check, download, warm up
    python tools/check_setup.py --smoke    # also make the robot say one line

Exits non-zero if anything critical failed.
"""

import argparse
import os
import platform
import sys
import time

import _bootstrap  # noqa: F401

CRITICAL = "critical"
WARNING = "warning"

# The vision model examples/09_vision.py defaults to. Separate from the chat
# model: it is the only example that needs it, so a missing one is a warning.
VISION_MODEL = "moondream"

results = []


def report(name, ok, detail="", level=CRITICAL):
    """Record and print one check result."""
    if ok:
        mark, label = "PASS", ""
    elif level == WARNING:
        mark, label = "WARN", ""
    else:
        mark, label = "FAIL", ""
    print(f"  [{mark}] {name}{label}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    results.append((name, ok, level))
    return ok


def section(title):
    print(f"\n{title}")


# -- checks ---------------------------------------------------------------


def check_python():
    v = sys.version_info
    ok = (3, 10) <= (v.major, v.minor) < (3, 14)
    in_venv = sys.prefix != sys.base_prefix
    detail = "" if ok else "kokoro-onnx needs >=3.10,<3.14. Recreate: uv venv --python 3.11"
    report(f"Python {v.major}.{v.minor}.{v.micro}", ok, detail)
    report(
        "Running inside the project venv",
        in_venv,
        "" if in_venv else "Run: source .venv/bin/activate",
    )
    return ok


def check_cwd():
    """ohbot resolves ohbotData/ relative to cwd, so this genuinely matters."""
    ok = os.path.exists("config.yaml") and os.path.isdir("ohbot_kit")
    return report(
        "Running from the project root",
        ok,
        "" if ok else "cd into the repo -- ohbotData/ is resolved relative to cwd",
    )


def check_robot():
    try:
        import serial
        import serial.tools.list_ports
    except ImportError as e:
        return report("Ohbot serial port", False, e)

    found = None
    for p in serial.tools.list_ports.comports():
        # Only macOS names its ports with "usb"; Windows uses COM3, so
        # applying this filter everywhere would skip every port and report
        # "no Ohbot" on a perfectly working Windows machine.
        if platform.system() == "Darwin" and "usb" not in p.device:
            continue
        try:
            with serial.Serial(p.device, 19200, timeout=0.5, write_timeout=0.5) as ser:
                ser.flushInput()
                ser.write(("v" + "\n").encode("latin-1"))
                reply = ser.readline()
            if b"v1" in reply or b"v2" in reply:
                found = "{} ({})".format(p.device, reply.decode("latin-1").strip())
                break
        except Exception:
            continue

    return report(
        "Ohbot responding on serial",
        found is not None,
        found or "Not found. Replug the USB-C cable (direct port, not a USB-A adapter).",
    )


def check_ollama(model, host="http://localhost:11434"):
    from ohbot_kit import llm

    try:
        models = llm.list_models(host)
    except llm.OllamaError as e:
        return report("Ollama running", False, e)

    report("Ollama running", True, f"{len(models)} model(s) installed")

    try:
        llm.check_model(model, host)
        return report(f"Chat model '{model}' installed", True)
    except llm.OllamaError as e:
        return report(f"Chat model '{model}' installed", False, e)


def download(url, dest):
    """Stream a file to dest with a progress line."""
    import requests

    tmp = dest + ".part"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
                done += len(block)
                if total:
                    print(
                        f"\r         {done * 100 / total:.0f}% of {total / 1e6:.0f} MB",
                        end="",
                        flush=True,
                    )
        print()
    os.replace(tmp, dest)  # atomic, so an interrupted download can't look complete


def check_kokoro():
    from ohbot_kit import tts

    for path in (tts.MODEL_FILE, tts.VOICES_FILE):
        if os.path.exists(path):
            continue
        name = os.path.basename(path)
        print(f"  [....] Downloading {name} ...")
        try:
            download(f"{tts.DOWNLOAD_BASE}/{name}", path)
        except Exception as e:
            return report("Kokoro model files", False, e)

    return report(
        "Kokoro model files present",
        tts.available(),
        "\n".join(
            f"{os.path.basename(p)}: {os.path.getsize(p) / 1e6:.0f} MB"
            for p in (tts.MODEL_FILE, tts.VOICES_FILE)
            if os.path.exists(p)
        ),
    )


def check_devices(cfg):
    """Resolve configured audio devices and report what they landed on."""
    from ohbot_kit import audio

    resolved = {}
    for key, kind in (("input_device", audio.INPUT), ("output_device", audio.OUTPUT)):
        name = cfg.get(f"audio.{key}")
        try:
            index = audio.resolve(name, kind)
        except audio.DeviceNotFound as e:
            report(f"Audio {kind}", False, e)
            resolved[kind] = None
            continue
        resolved[kind] = index
        report(
            f"Audio {kind}",
            True,
            "{}{}".format(
                audio.device_name(index),
                "" if name else f"  (system default -- set audio.{key} to pin it)",
            ),
        )
    return resolved


def check_microphone(device=None):
    """A warning, not a failure -- typed mode is a perfectly good demo.

    Checks the *configured* device, not the system default: otherwise this can
    pass while the demo fails on a different mic.
    """
    try:
        from ohbot_kit import voice
    except ImportError as e:
        return report("Microphone", False, e, level=WARNING)

    try:
        listener = voice.Listener.__new__(voice.Listener)  # skip the model load
        listener.device = device
        listener.noise_multiplier = voice.NOISE_MULTIPLIER
        listener.threshold = voice.FLOOR
        listener.calibrate()
        return report("Microphone delivering audio", True, level=WARNING)
    except voice.MicrophoneBlocked as e:
        return report("Microphone delivering audio", False, e, level=WARNING)
    except Exception as e:
        return report("Microphone delivering audio", False, e, level=WARNING)


def check_camera(index=0):
    """A warning, not a failure -- only 09_vision.py needs a camera.

    Worth checking here because macOS denies camera access to the terminal
    silently: OpenCV reports "not authorized to capture video" and then a
    camera that simply won't open, which reads as broken hardware rather than a
    permission you have to grant. Finding that out now beats finding it out
    while an audience watches a robot describe nothing.
    """
    try:
        import cv2
    except ImportError:
        return report(
            "OpenCV installed",
            False,
            "Only needed for 09_vision.py. It is in requirements.txt:\n"
            "uv pip install -r requirements.txt",
            level=WARNING,
        )

    camera = cv2.VideoCapture(index)
    try:
        if not camera.isOpened():
            return report(
                f"Camera {index} opens",
                False,
                "On macOS, grant camera access to the app running this -- System Settings\n"
                "> Privacy & Security > Camera -- then restart it. Otherwise check no other\n"
                "program is holding the camera.",
                level=WARNING,
            )
        # The first frames after opening are often unexposed, which is why
        # 09_vision.py discards a few before its first look. Do the same here,
        # or this can report a working camera as delivering a black frame.
        ok, frame = False, None
        for _ in range(5):
            ok, frame = camera.read()
        detail = ""
        if ok and frame is not None:
            detail = f"{frame.shape[1]}x{frame.shape[0]}"
        return report(
            "Camera delivering frames",
            ok,
            detail or "Camera opened but returned no frame.",
            level=WARNING,
        )
    finally:
        camera.release()


def check_vision_model(model=VISION_MODEL, host="http://localhost:11434"):
    """Also a warning: every example except 09_vision.py works without this."""
    from ohbot_kit import llm

    try:
        llm.check_model(model, host)
        return report(f"Vision model '{model}' installed", True, level=WARNING)
    except llm.OllamaError:
        return report(
            f"Vision model '{model}' installed",
            False,
            f"Only needed for 09_vision.py:  ollama pull {model}",
            level=WARNING,
        )


# -- warmups --------------------------------------------------------------


def warm_ollama(model):
    from ohbot_kit import llm

    t0 = time.time()
    llm.Conversation(model=model).warm_up()
    report("Ollama model loaded", True, f"{time.time() - t0:.1f}s")


def warm_kokoro(voice_name):
    from ohbot_kit import tts

    try:
        t0 = time.time()
        engine = tts.KokoroTTS(voice=voice_name)
        engine.synth("Warming up.")
        return report(
            f"Kokoro loaded (voice: {voice_name})",
            True,
            f"{time.time() - t0:.1f}s, {len(engine.voices())} voices available",
        )
    except Exception as e:
        return report("Kokoro loaded", False, e)


def warm_whisper():
    try:
        import numpy as np

        from ohbot_kit import voice

        t0 = time.time()
        listener = voice.Listener()
        # Transcribe a second of silence purely to force the model to load.
        listener.transcribe(np.zeros(16000, dtype=np.float32))
        return report(
            f"Whisper loaded ({voice.MODEL_SIZE})",
            True,
            f"{time.time() - t0:.1f}s",
        )
    except Exception as e:
        return report("Whisper loaded", False, e, level=WARNING)


def smoke_test(voice_name, output_device=None):
    """Opt-in: prove the whole chain by making the robot actually speak."""
    from ohbot_kit import audio, tts
    from ohbot_kit.robot import Ohbot

    try:
        audio.install_output(output_device)
        tts.install(tts.KokoroTTS(voice=voice_name))
    except Exception as e:
        return report("Smoke test", False, e)

    try:
        with Ohbot(idle=False) as bot:
            bot.speak("Setup complete. I am ready for the demo.")
        return report("Smoke test: robot spoke and moved", True)
    except Exception as e:
        return report("Smoke test", False, e)


# -- main -----------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Pre-demo setup and checks.")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument("--model", default=None, help="Ollama chat model")
    p.add_argument("--voice-name", default=None, help="Kokoro voice")
    p.add_argument("--smoke", action="store_true", help="also make the robot speak")
    p.add_argument("--skip-mic", action="store_true", help="skip the microphone check")
    p.add_argument("--skip-camera", action="store_true", help="skip the camera check")
    p.add_argument("--camera", type=int, default=0, help="camera index for the check")
    args = p.parse_args()

    print("Ohbot demo setup\n" + "=" * 40)

    from ohbot_kit import config as config_mod

    cfg = config_mod.load(args.config)
    model = args.model or cfg.get("llm.model", "phi4-mini")
    voice_name = args.voice_name or cfg.get("tts.voice", "af_heart")

    section("Environment")
    check_python()
    check_cwd()
    report(
        "Config loaded",
        True,
        ", ".join(cfg.sources) if cfg.sources else "none found -- using code defaults",
    )

    section("Hardware and services")
    check_robot()
    check_ollama(model, cfg.get("llm.host", "http://localhost:11434"))

    section("Audio devices")
    resolved = check_devices(cfg)

    section("Models")
    check_kokoro()

    section("Warming up (so the first live response isn't slow)")
    warm_ollama(model)
    warm_kokoro(voice_name)
    warm_whisper()

    if not args.skip_mic:
        section("Microphone (only needed for --voice)")
        check_microphone(resolved.get("input"))

    if not args.skip_camera:
        section("Camera and vision (only needed for 09_vision.py)")
        check_camera(args.camera)
        check_vision_model(VISION_MODEL, cfg.get("llm.host", "http://localhost:11434"))

    if args.smoke:
        section("Smoke test")
        smoke_test(voice_name, resolved.get("output"))

    # -- summary
    failed = [n for n, ok, lvl in results if not ok and lvl == CRITICAL]
    warned = [n for n, ok, lvl in results if not ok and lvl == WARNING]

    print("\n" + "=" * 40)
    if failed:
        print(f"NOT READY -- {len(failed)} critical problem(s):")
        for n in failed:
            print(f"  - {n}")
        return 1

    if warned:
        print(f"READY (with {len(warned)} warning(s)):")
        for n in warned:
            print(f"  - {n}")
        # Name the mode each warning actually costs you. A blanket "voice mode
        # will not work" is wrong when the only thing missing is a webcam.
        print("\nTyped mode will work.")
        if any("Microphone" in n or "Whisper" in n for n in warned):
            print("Voice mode (--voice) will not.")
        if any(k in n for n in warned for k in ("Camera", "Vision", "OpenCV")):
            print("examples/09_vision.py will not.")
    else:
        print("READY. Everything checks out.")

    print("\nNext:  python examples/01_hello_robot.py     then  python examples/08_empathy_chat.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
