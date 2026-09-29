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
import audio
import reminder as r
import time as time_module

from ohbot_kit import Ohbot, expression, setup

def takeInAudio():
    pass

def respondToUser(userTranscript, bot, convo):
    # text = input("You: ").strip()
    # text = "hello! how are you today?"
    text = (userTranscript or "").strip()
    if not text or text.lower() in ("never mind", "quit", "exit"):
        return None

    bot.set_state("thinking")
    action = convo.respond_with_action(text, expression.EMOTIONS, expression.GESTURE_NAMES)
    print("Ohbot: {}".format(action["say"]))
    bot.set_state("speaking")
    bot.speak(action["say"], emotion=action["emotion"], gesture=action["gesture"])
    return action["say"]


def checkForReminder():
    now = time_module.time()
    for reminder_time in pull_time():
        if float(reminder_time) <= now:
            reminder_entry = pull_reminder(reminder_time)
            if reminder_entry:
                return reminder_entry
    return None


def pull_reminder(reminder_time):
    for reminder in r.reminders:
        if reminder.get("time") == reminder_time:
            return reminder
    return None


def pull_time():
    times = []
    for reminder in r.reminders:
        reminder_time = reminder.get("time")
        if reminder_time is not None:
            times.append(reminder_time)
    return times


def reminderCode():
    pass

def main():
    # cfg           : settings from config.yaml (+ config.local.yaml)
    # convo         : the LLM, with your persona's prompt and voice
    # robot_kwargs  : idle motion, eye colours, timings -- pass to Ohbot()
    cfg, convo, robot_kwargs = setup()

    # `with` guarantees the motors are detached at the end, even on a crash.
    # Without it they stay attached, drawing current and buzzing.
    with Ohbot(**robot_kwargs) as bot:
        bot.set_state("loading")
        convo.warm_up()
        bot.set_state("ready")
        bot.recentre()
        bot.speak("Hello, I am Ohbot.", emotion="happy", gesture="perk_up")
        bot.express("playful")
        bot.gesture("tiny_wave", blocking=True)
        bot.speak("Let me know if you need me to do anything for you!", emotion="happy", gesture="perk_up")

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

        running = True
        while running:
            touchSensor = bot.read_sensor(3) > 7
            reminder_entry = checkForReminder()

            if touchSensor:
                bot.set_state("listening")
                bot.listening(True)
                bot.gesture("double_blink", blocking=True)
                userTranscript = audio.record_audio(5)
                bot.listening(False)
                print(userTranscript)
                if userTranscript:
                    respondToUser(userTranscript, bot, convo)
                bot.set_state("ready")

            if reminder_entry:
                reminderCode()
                print("Reminder triggered:", reminder_entry.get("text"))
                bot.speak("There's a reminder for you!")
                bot.speak(reminder_entry.get("text"))
                r.remove_reminder(reminder_entry.get("text"))

            time_module.sleep(0.2)

        bot.set_state("ready")
        bot.speak("Goodbye!", emotion="happy", gesture="nod")
        r.close()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print()