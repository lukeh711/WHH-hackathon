"""COPY THIS FILE to start your project.

    cp template.py my_project.py
    python my_project.py

Everything is wired up already: config, the local LLM, the Kokoro voice, audio
device routing, expression and gestures. Write your idea in main().

Reference:
  docs/API.md            every motor, pose and gesture
  docs/CHALLENGES.md     project ideas if you're still deciding
  examples/              worked examples, 01 through 09
"""

import sys

from ohbot_kit import Ohbot, expression, setup


def main():
    # cfg           : settings from config.yaml (+ config.local.yaml)
    # convo         : the LLM, with your persona's prompt and voice
    # robot_kwargs  : idle motion, eye colours, timings -- pass to Ohbot()
    cfg, convo, robot_kwargs = setup()
    convo.warm_up()
    bot.recentre()

    # `with` guarantees the motors are detached at the end, even on a crash.
    # Without it they stay attached, drawing current and buzzing.
    with Ohbot(**robot_kwargs) as bot:
        bot.speak("Hello, I am Ohbot.", emotion="happy", gesture="perk_up")

        # ------------------------------------------------------------------
        # YOUR CODE HERE
        #
        #   bot.speak(text, emotion=..., gesture=...)   speak, with a face and
        #                                               a movement underneath
        #   bot.express("curious")                      hold a face
        #   bot.gesture("slow_nod")                     movement on its own
        #   bot.look_at("up_left")                      eyes + head together
        #   bot.listening(True)                         nod while the user talks
        #   bot.read_sensor(0)                          0.0-10.0
        #   bot.set_eyes(0, 10, 0)                      r, g, b, each 0-10
        #
        #   convo.stream_sentences(text)                plain reply, streamed
        #   convo.respond_with_action(text,             reply + emotion + gesture
        #       expression.EMOTIONS, expression.GESTURE_NAMES)
        #
        # Add your own entries to expression.POSES and expression.GESTURES --
        # they automatically become choices the LLM can pick.
        # ------------------------------------------------------------------

        while True:
            text = input("You: ").strip()
            if not text or text in ("/quit", "/exit"):
                break

            action = convo.respond_with_action(text, expression.EMOTIONS, expression.GESTURE_NAMES)
            print("Ohbot: {}".format(action["say"]))
            bot.speak(action["say"], emotion=action["emotion"], gesture=action["gesture"])

        bot.speak("Goodbye!", emotion="happy", gesture="nod")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print()