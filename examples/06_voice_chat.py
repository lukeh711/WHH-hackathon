"""Talk to the robot with your voice.

    python examples/06_voice_chat.py

Speech recognition runs locally with faster-whisper -- no internet, no API key,
nothing leaves the laptop.

TWO THINGS THAT CATCH PEOPLE OUT

1. macOS must grant microphone access to your terminal:
   System Settings > Privacy & Security > Microphone. If the mic returns pure
   silence the program says so and exits, rather than waiting forever for a
   sound that can never arrive.

2. The robot must not hear itself. record_utterance() is given `bot` so it can
   discard audio while the robot is speaking. With laptop speakers it would
   otherwise transcribe its own voice and reply to itself. A headset avoids
   the problem entirely -- and in a room full of robots, it's the only thing
   that stops yours hearing everyone else's.
"""

import _bootstrap  # noqa: F401

from ohbot_kit import Ohbot, make_listener, setup
from ohbot_kit.voice import MicrophoneBlocked

cfg, convo, robot_kwargs = setup()
listener = make_listener(cfg)
convo.warm_up()

with Ohbot(**robot_kwargs) as bot:
    bot.speak("Hello! I am listening.", emotion="happy", gesture="perk_up")
    try:
        while True:
            bot.listening(True)  # nod and blink while you talk
            audio_in = listener.record_utterance(bot)
            if audio_in is None:
                continue
            text = listener.transcribe(audio_in)
            if not text:
                continue
            print(f"You: {text}")
            bot.listening(False)

            bot.set_state("thinking")
            for sentence in convo.stream_sentences(text):
                print(f"Ohbot: {sentence}")
                bot.speak(sentence)
    except MicrophoneBlocked as e:
        print(f"\n[microphone] {e}")
    except KeyboardInterrupt:
        print()
    bot.speak("Goodbye!")
