"""Speech: voices, speed, and how lip sync actually works.

    python examples/03_speech.py
    python examples/03_speech.py --list-voices

Kokoro-82M gives 54 voices and is FASTER than the macOS `say` voice
(0.62s to generate 4.5s of audio, versus 1.25s), so there's no latency cost to
the better sound.
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from ohbot_kit import Ohbot, setup, tts

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--list-voices", action="store_true")
p.add_argument("--voice", default=None, help="try one specific voice")
args = p.parse_args()

if args.list_voices:
    print("\n".join(tts.KokoroTTS().voices()))
    sys.exit(0)

# A few contrasting voices. Full list with --list-voices.
VOICES = [args.voice] if args.voice else ["af_heart", "bf_emma", "am_michael", "bm_george"]

cfg, _convo, robot_kwargs = setup()

with Ohbot(**robot_kwargs) as bot:
    for name in VOICES:
        try:
            tts.install(tts.KokoroTTS(voice=name))
        except Exception as e:
            print(f"skipping {name}: {e}")
            continue
        print("voice:", name)
        bot.speak(f"Hello, my name is Ohbot and this is the {name} voice.")

    # Speed is a Kokoro parameter, not a playback trick -- the speech is
    # generated at that pace, so it still sounds natural.
    for speed in (0.8, 1.3):
        tts.install(tts.KokoroTTS(voice=VOICES[0], speed=speed))
        print("speed:", speed)
        bot.speak(f"This is me talking at speed {speed}.")

    # WHY THE LIP MOVES: ohbot.say() does NOT use phonemes on macOS. It reads
    # the generated WAV, sums the sample bytes in chunks of framerate/10, and
    # drives the lip from that loudness envelope. So any audio file gets lip
    # sync for free -- which is exactly how Kokoro was swapped in.
    tts.install(tts.KokoroTTS(voice=VOICES[0]))
    bot.speak("My lip follows how loud the audio is, not the sounds themselves.")
