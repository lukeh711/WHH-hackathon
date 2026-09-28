"""The smallest possible talking robot: Ollama in, speech out.

    python examples/04_chat_basic.py

Deliberately plain -- no gestures, no expression. Compare it with
08_empathy_chat.py to see exactly what the empathy layer adds.

Replies are streamed and spoken sentence by sentence, so the robot starts
talking after the first sentence instead of waiting for the whole answer.
"""

import _bootstrap  # noqa: F401

from ohbot_kit import Ohbot, OllamaError, setup

cfg, convo, robot_kwargs = setup()
convo.warm_up()  # load the model now, so the first real reply isn't slow

print("Type to chat. Ctrl-C or /quit to stop.\n")

with Ohbot(**robot_kwargs) as bot:
    bot.speak("Hello! Ask me anything.")
    try:
        while True:
            text = input("You: ").strip()
            if not text or text in ("/quit", "/exit"):
                break

            bot.set_state("thinking")
            print("Ohbot: ", end="", flush=True)
            try:
                for sentence in convo.stream_sentences(text):
                    print(sentence, end=" ", flush=True)
                    bot.speak(sentence)
            except OllamaError as e:
                print(f"\n[error] {e}")
            print()
            bot.set_state("listening")
    except (KeyboardInterrupt, EOFError):
        print()
    bot.speak("Bye!")
