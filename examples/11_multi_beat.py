"""Expression that changes mid-reply.

    python examples/11_multi_beat.py

Compare with 08_empathy_chat.py, which picks ONE face for a whole reply. Here
the LLM breaks its answer into beats, each with its own emotion, so the robot
can be concerned while it acknowledges your problem and brighter as it offers
help:

    [sad     / slow_nod] "I'm really sorry to hear that."
    [excited / perk_up ] "But don't worry, practice makes perfect!"

That shift is most of what makes a longer reply feel meant rather than recited.

Try: "I failed my driving test today, third time."

TWO THINGS TO KNOW

Each beat is a separate say() call, so a three-beat reply takes noticeably
longer to deliver than one sentence. Keep max_beats low for live demos.

A small model asked for a list sometimes returns one overlong beat, or an empty
list. respond_with_beats validates and truncates rather than trusting it, so
you get a usable list or an OllamaError -- never a half-formed one.
"""

import sys

import _bootstrap  # noqa: F401

from ohbot_kit import Ohbot, OllamaError, expression, setup

MAX_BEATS = 3

cfg, convo, robot_kwargs = setup()
convo.warm_up()

print("Multi-beat chat. Try something with a turn in it. Ctrl-C to stop.\n")

with Ohbot(**robot_kwargs) as bot:
    bot.speak("Tell me how your day went.", emotion="curious", gesture="lean_in")

    try:
        while True:
            text = input("You: ").strip()
            if not text or text in ("/quit", "/exit"):
                break

            bot.set_state("thinking")
            bot.express("thinking")

            try:
                beats = convo.respond_with_beats(text, expression.EMOTIONS, MAX_BEATS)
            except OllamaError as e:
                print(f"[llm] {e}", file=sys.stderr)
                continue

            for beat in beats:
                print(f"  [{beat['emotion']:<12} {beat['gesture']:<10}] {beat['say']}")
                bot.speak(beat["say"], emotion=beat["emotion"], gesture=beat["gesture"])

            bot.set_state("listening")
    except (KeyboardInterrupt, EOFError):
        print()

    bot.speak("Talk soon!", emotion="happy", gesture="nod")
