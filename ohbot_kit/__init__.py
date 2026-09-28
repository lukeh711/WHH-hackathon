"""Ohbot hackathon kit.

Everything you normally need, from one import:

    from ohbot_kit import Ohbot, Conversation, setup

    cfg, convo, bot_kwargs = setup()          # config, LLM, audio, voice
    with Ohbot(**bot_kwargs) as bot:
        bot.speak("Hello!", emotion="happy", gesture="nod")

The submodules are still there when you want them:

    from ohbot_kit import audio, call, expression, tts, voice

`voice` and `call` are deliberately not re-exported above: importing them pulls
in the speech model stack, which is pure cost for a typed chat. Import them
directly when you need them.

IMPORTANT: run your scripts from the repository root. The ohbot library
resolves its ohbotData/ folder (motor calibration, speech database, sounds)
relative to the working directory, so running from elsewhere silently creates a
second, uncalibrated copy.
"""

from __future__ import annotations

from typing import Any

from . import audio, config, expression, llm, serial_safe, tts
from .config import Config
from .config import load as load_config
from .expression import EMOTIONS, GESTURE_NAMES, GESTURES, LOOK_DIRECTIONS, POSES
from .llm import Conversation, OllamaError
from .robot import Ohbot
from .serial_safe import install_write_lock
from .tts import KokoroTTS

__all__ = [
    "Ohbot",
    "Conversation",
    "OllamaError",
    "KokoroTTS",
    "Config",
    "load_config",
    "setup",
    "POSES",
    "GESTURES",
    "EMOTIONS",
    "GESTURE_NAMES",
    "LOOK_DIRECTIONS",
    "install_write_lock",
    "audio",
    "config",
    "expression",
    "llm",
    "tts",
]


def setup(
    model: str | None = None,
    persona: str | None = None,
    voice_name: str | None = None,
    config_path: str = "config.yaml",
) -> tuple[Config, Conversation, dict[str, Any]]:
    """Do the boilerplate: load config, wire audio and speech, build the LLM.

    Returns (cfg, conversation, robot_kwargs). Pass robot_kwargs straight into
    Ohbot(). Every example starts this way, so the interesting code isn't
    buried under twenty lines of setup.

    Nothing here needs a robot attached -- the library no-ops its motor writes
    when nothing is connected, so you can build the whole loop without hardware.
    """
    cfg = config.load(config_path, warn=False)

    active_persona = cfg.persona(persona)
    chat_model = model or cfg.get("llm.model", llm.DEFAULT_MODEL)
    host = cfg.get("llm.host", llm.HOST)

    # Route speech to the configured output device. ohbot always plays through
    # the system default otherwise -- see audio.install_output.
    try:
        audio.install_output(audio.resolve(cfg.get("audio.output_device"), audio.OUTPUT))
    except audio.DeviceNotFound as e:
        print(f"[audio] {e}")

    # Kokoro sounds better and is faster than macOS `say`, but a missing model
    # file should degrade the demo, not stop it.
    chosen_voice = voice_name or active_persona.get("voice") or cfg.get("tts.voice")
    if cfg.get("tts.engine", "kokoro") == "kokoro":
        try:
            tts.install(
                tts.KokoroTTS(
                    voice=chosen_voice or tts.DEFAULT_VOICE,
                    speed=cfg.get("tts.speed", tts.DEFAULT_SPEED),
                )
            )
        except Exception as e:
            print(f"[tts] using macOS say: {e}")

    conversation = llm.Conversation(
        model=chat_model,
        system=active_persona.get("system_prompt", llm.SYSTEM_PROMPT),
        host=host,
        temperature=cfg.get("llm.temperature", 0.7),
        num_predict=cfg.get("llm.num_predict", 80),
    )

    robot_kwargs = {
        "idle": cfg.get("robot.idle", True),
        "colours": cfg.get("robot.eye_colours"),
        "blink_interval": cfg.get("robot.blink_interval", (2, 6)),
        "drift_interval": cfg.get("robot.drift_interval", (4, 9)),
        "has_headroll": cfg.get("robot.has_headroll", True),
    }
    return cfg, conversation, robot_kwargs


def make_listener(cfg: Config) -> Any:
    """Build a microphone listener from config. Imported lazily -- the speech
    model is heavy and only needed when you actually want voice input."""
    from . import voice

    return voice.Listener(
        device=audio.resolve(cfg.get("audio.input_device"), audio.INPUT),
        model_size=cfg.get("speech_to_text.model", voice.MODEL_SIZE),
        silence_seconds=cfg.get("speech_to_text.silence_seconds", voice.SILENCE_SECONDS),
        min_speech_seconds=cfg.get("speech_to_text.min_speech_seconds", voice.MIN_SPEECH_SECONDS),
        noise_multiplier=cfg.get("speech_to_text.noise_multiplier", voice.NOISE_MULTIPLIER),
    )
