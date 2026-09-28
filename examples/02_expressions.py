"""The visual catalogue: every pose, every gesture, and coordinated looking.

    python examples/02_expressions.py            # everything
    python examples/02_expressions.py poses      # just the faces
    python examples/02_expressions.py gestures   # just the movements
    python examples/02_expressions.py looks      # just the gaze directions

Each item is announced so you can tell them apart. Use this to decide what to
remix -- add to expression.POSES or expression.GESTURES and it appears here,
and becomes a choice the LLM can make, automatically.
"""

import sys
import time

import _bootstrap  # noqa: F401

from ohbot_kit import audio, tts
from ohbot_kit import config as config_mod
from ohbot_kit import expression as ex
from ohbot_kit.robot import Ohbot


def setup_audio(cfg):
    try:
        audio.install_output(audio.resolve(cfg.get("audio.output_device"), audio.OUTPUT))
        tts.install(tts.KokoroTTS(voice=cfg.get("tts.voice", "af_heart")))
    except Exception as e:
        print(f"[tts] falling back to macOS say: {e}")


def show_poses(bot):
    print("\nPOSES -- held facial expressions\n")
    for name in ex.EMOTIONS:
        print(f"  {name}")
        bot.express("neutral")
        time.sleep(0.6)
        bot.express(name)
        bot.speak(f"This is {name}.", recentre=False)
        time.sleep(1.0)
    bot.express("neutral")
    bot.recentre()


def show_gestures(bot):
    print("\nGESTURES -- movements, played underneath speech\n")
    for name in ex.GESTURE_NAMES:
        meaning = ex.GESTURE_MEANINGS.get(name, "")
        print(f"  {name:<14} {meaning}")
        bot.express("neutral")
        bot.recentre()
        time.sleep(0.8)
        # Spoken WITH the gesture, which is the point -- the movement runs
        # under the audio rather than before it.
        bot.speak("{}. {}.".format(name.replace("_", " "), meaning), gesture=name)
        time.sleep(0.5)


def show_looks(bot):
    print("\nLOOKING -- eyes lead, head follows partway\n")
    for name in [
        "up_left",
        "up",
        "up_right",
        "right",
        "down_right",
        "down",
        "down_left",
        "left",
        "user",
    ]:
        x, y = ex.LOOK_DIRECTIONS[name]
        print(f"  {name:<11} (x={x}, y={y})")
        bot.look_at(name)
        time.sleep(1.4)

    print("\n  eyes only, head still")
    for name in ("left", "right", "user"):
        bot.look_at(name, with_head=False)
        time.sleep(1.0)

    bot.look_at("user")
    bot.speak("And back to looking at you.")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = config_mod.load(warn=False)
    setup_audio(cfg)

    # Idle motion off: it would add drift on top of what you're trying to see.
    with Ohbot(idle=False) as bot:
        bot.speak("Here is everything I can do with my face.")
        if which in ("all", "poses"):
            show_poses(bot)
        if which in ("all", "gestures"):
            show_gestures(bot)
        if which in ("all", "looks"):
            show_looks(bot)
        bot.express("happy")
        bot.speak("That is the whole catalogue.", gesture="nod")

    return 0


if __name__ == "__main__":
    sys.exit(main())
