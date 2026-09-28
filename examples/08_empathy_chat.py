"""FLAGSHIP: a robot that reacts, not just replies.

The difference between this and 04_chat_basic.py is that the LLM chooses a
face and a movement alongside its words, and the movement plays WHILE the robot
speaks. It also nods and blinks while YOU are talking, so it looks like it is
listening rather than waiting.

    python examples/08_empathy_chat.py
    python examples/08_empathy_chat.py --voice     # talk to it

Try: "I had a really rough day." -- watch the face, not just the words.

HOW IT WORKS

The LLM returns JSON constrained by a schema (Ollama's `format` field):

    {"say": "...", "emotion": "sympathetic", "gesture": "slow_nod",
     "gaze_x": 5, "gaze_y": 4}

We use that rather than the tools API because phi4-mini advertises tool support
but never actually emits tool_calls -- it just refuses. Schema-constrained JSON
works first time, every time, in well under a second.

Remix ideas: add poses to expression.POSES, add keyframe sequences to
expression.GESTURES -- both automatically become choices the LLM can pick,
because the schema is built from those tables.
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from ohbot_kit import config as config_mod
from ohbot_kit import expression, llm, tts
from ohbot_kit.robot import Ohbot


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--voice", action="store_true", help="talk instead of typing")
    p.add_argument("--model", default=None)
    args = p.parse_args()

    cfg = config_mod.load(warn=False)
    model = args.model or cfg.get("llm.model", llm.DEFAULT_MODEL)
    persona = cfg.persona()

    try:
        llm.check_model(model, cfg.get("llm.host", llm.HOST))
    except llm.OllamaError as e:
        print(e, file=sys.stderr)
        return 1

    convo = llm.Conversation(
        model=model,
        system=persona.get("system_prompt", llm.SYSTEM_PROMPT),
        host=cfg.get("llm.host", llm.HOST),
    )

    try:
        tts.install(tts.KokoroTTS(voice=persona.get("voice", tts.DEFAULT_VOICE)))
    except Exception as e:
        print(f"[tts] using macOS say: {e}", file=sys.stderr)

    listener = None
    if args.voice:
        from ohbot_kit import audio, voice

        listener = voice.Listener(device=audio.resolve(cfg.get("audio.input_device"), audio.INPUT))

    print("Empathy chat. Say something with feeling in it. Ctrl-C to stop.\n")
    convo.warm_up()

    with Ohbot(idle=True) as bot:
        bot.speak("Hello! How are you doing today?", emotion="happy", gesture="perk_up")

        try:
            while True:
                # Backchannel: nod and blink while the user is talking.
                bot.listening(True)
                if listener:
                    audio_in = listener.record_utterance(bot)
                    if audio_in is None:
                        continue
                    text = listener.transcribe(audio_in)
                    if not text:
                        continue
                    print(f"You: {text}")
                else:
                    text = input("You: ").strip()
                bot.listening(False)

                if not text or text in ("/quit", "/exit"):
                    break

                bot.set_state("thinking")
                bot.express("thinking")

                try:
                    action = convo.respond_with_action(
                        text, expression.EMOTIONS, expression.GESTURE_NAMES
                    )
                except llm.OllamaError as e:
                    print(f"[llm] {e}", file=sys.stderr)
                    continue

                print(
                    "Ohbot [{}/{}]: {}".format(action["emotion"], action["gesture"], action["say"])
                )

                bot.gaze(action.get("gaze_x", 5), action.get("gaze_y", 5))
                # Face + movement + voice together -- the gesture runs
                # underneath the audio rather than before it.
                bot.speak(
                    action["say"],
                    emotion=action["emotion"],
                    gesture=action["gesture"],
                )
        except (KeyboardInterrupt, EOFError):
            print()

        bot.speak("Talk to you later!", emotion="happy", gesture="nod")

    return 0


if __name__ == "__main__":
    sys.exit(main())
