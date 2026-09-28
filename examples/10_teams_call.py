"""Ohbot joins a Teams call, listens, and answers when spoken to.

    python examples/10_teams_call.py

Set the audio routing up FIRST -- all of it in docs/TEAMS.md. Then check the cable
is actually carrying audio:

    python tools/check_call_audio.py

WHAT IS CONFIRMED WORKING

Teams speaker BlackHole 2ch, Teams mic BlackHole 16ch, noise suppression off, and
YOU on a second device -- your phone, joined to the same meeting. Then:

    python examples/10_teams_call.py --input "BlackHole 2ch" --output "BlackHole 16ch"

Being on the call from the same Mac as the robot is WIP: Teams has one microphone
slot and the robot occupies it. See docs/TEAMS.md, "Being on the call too".

Teams has no bot to install and does not need one. Teams only cares which audio
devices it is pointed at, so Ohbot becomes the microphone and Teams' speaker
becomes Ohbot's ears. Works identically on Zoom, Meet, Slack huddles.

THE ONE THING THAT MAKES THIS USABLE

It waits to be addressed. The other examples reply to every utterance, which is
fine at a desk and unbearable on a call -- the robot talks over people
constantly, a second behind, and someone hangs up. So nothing is spoken unless
the utterance starts with the robot's name:

    "So I think we should ship on Friday."   -> nods along, says nothing
    "Ohbot, what day is it?"                 -> answers out loud into the call

Pass --open-mic to remove the gate, which is worth doing once to see why the
gate exists.

IT ALSO DOES NOT SAY HELLO

Speaking unprompted into a live call is rude, and every greeting lands after
someone else has started talking. It perks up on the terminal instead.

Remix ideas: swap the wake word with --wake; have it answer only questions
(text ending in "?"); keep a running note of action items and read them back
when asked; make it look towards whoever is speaking using gaze.
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from ohbot_kit import audio, expression, llm, setup
from ohbot_kit.call import WAKE_WORDS, make_call_listener, wake_word
from ohbot_kit.robot import Ohbot


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=None, help="Ollama model to use")
    p.add_argument(
        "--wake",
        default=None,
        help="word that addresses the robot (default: ohbot, hey ohbot, ...)",
    )
    p.add_argument(
        "--open-mic",
        action="store_true",
        help="reply to everything, with no wake word -- try it once",
    )
    p.add_argument("--list-devices", action="store_true", help="print devices and exit")
    # Override the routing without editing config.local.yaml, so you can try a
    # setup before committing to it. Most of getting this working is trying things.
    p.add_argument("--input", default=None, help='ears, e.g. "BlackHole 2ch"')
    p.add_argument("--output", default=None, help='voice, e.g. "MacBook Pro Speakers"')
    args = p.parse_args()

    if args.list_devices:
        print(audio.describe())
        return 0

    cfg, convo, robot_kwargs = setup(model=args.model)
    words = (args.wake.lower(),) if args.wake else WAKE_WORDS

    try:
        if args.output:
            audio.install_output(audio.resolve(args.output, audio.OUTPUT))
        in_device = audio.resolve(args.input, audio.INPUT) if args.input else None
        listener = make_call_listener(cfg, device=in_device)
    except audio.DeviceNotFound as e:
        print(e, file=sys.stderr)
        print("\nSee docs/TEAMS.md for the routing this example expects.", file=sys.stderr)
        return 1

    print(f"Ears : {listener.describe()}")
    try:
        # setup() already reported a bad name and fell back to the default.
        out_name = audio.device_name(
            audio.resolve(args.output or cfg.get("audio.output_device"), audio.OUTPUT)
        )
    except audio.DeviceNotFound:
        out_name = "system default"
    print(f"Voice: {out_name}")
    print(
        "\nOn the call, "
        + (
            "replying to everything (--open-mic)."
            if args.open_mic
            else f'say "{words[-1]}, ..." to be answered.'
        )
        + "\nTell the other participants a robot is listening. Ctrl-C to leave.\n"
    )
    convo.warm_up()

    unaddressed = 0

    with Ohbot(**robot_kwargs) as bot:
        bot.express("curious")  # awake, but silent -- no greeting
        listener.calibrate()

        try:
            while True:
                # Nod and blink while other people talk: on a call this is the
                # entire performance most of the time, since it stays quiet.
                bot.listening(True)

                heard = listener.record_utterance(bot)
                if heard is None:
                    continue
                text = listener.transcribe(heard)
                if not text:
                    continue

                if args.open_mic:
                    request = text
                else:
                    request = wake_word(text, words)
                    if request is None:
                        # Staying silent here is the correct behaviour, and it is
                        # indistinguishable from being broken. Say why, twice,
                        # then stop cluttering the transcript.
                        unaddressed += 1
                        if unaddressed <= 2:
                            print(
                                f"call: {text}\n"
                                f'      ^ heard, but not addressed. Say "{words[-1]}, '
                                'what day is it?" to get an answer,\n'
                                "        or restart with --open-mic to reply to everything. "
                                "If the name above\n"
                                "        came out mangled, use --wake with a word Whisper "
                                "hears reliably."
                            )
                        else:
                            print(f"call: {text}")
                        continue  # not for us; keep listening
                    if not request:
                        request = "Someone said your name. Ask what they need."

                print(f"call: {text}  <- addressed to Ohbot")
                bot.listening(False)
                bot.set_state("thinking")
                bot.express("thinking")

                try:
                    action = convo.respond_with_action(
                        request, expression.EMOTIONS, expression.GESTURE_NAMES
                    )
                except llm.OllamaError as e:
                    print(f"[llm] {e}", file=sys.stderr)
                    continue

                print(
                    "Ohbot [{}/{}]: {}".format(action["emotion"], action["gesture"], action["say"])
                )

                bot.gaze(action.get("gaze_x", 5), action.get("gaze_y", 5))
                # Speaks into the call and the room at once: its output device
                # is a Multi-Output containing both. See docs/TEAMS.md.
                bot.speak(
                    action["say"],
                    emotion=action["emotion"],
                    gesture=action["gesture"],
                )
        except KeyboardInterrupt:
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
