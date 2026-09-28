"""START HERE: move the robot and make it talk.

    python examples/01_hello_robot.py

The one rule worth learning first is the try/finally. If a script dies before
close(), the motors stay attached -- drawing current and buzzing audibly -- until
you replug the robot. The `with` block below does that for you.
"""

import time

import _bootstrap  # noqa: F401

from ohbot_kit import Ohbot, setup

# Loads config.yaml, wires up audio and the Kokoro voice. No robot needed for
# this to succeed -- motor commands are silently ignored when nothing is
# plugged in, so you can run every example without hardware.
cfg, _convo, robot_kwargs = setup()

# `with` guarantees close() runs, even if something below raises.
with Ohbot(**robot_kwargs) as bot:
    # Motors take a position 0-10 and an optional speed 0-10.
    bot.express("happy")
    bot.speak("Hello! I am Ohbot.")

    # Look around: eyes lead, the head follows partway behind.
    for direction in ("left", "right", "up", "user"):
        bot.look_at(direction)
        time.sleep(1.0)

    # A gesture passed to speak() plays UNDERNEATH the speech rather than
    # before it -- that overlap is what makes it look alive.
    bot.speak("I can move while I talk.", emotion="excited", gesture="nod")

    # Eye colour is (r, g, b), each 0-10.
    for r, g, b in [(10, 0, 0), (0, 10, 0), (0, 0, 10)]:
        bot.set_eyes(r, g, b)
        time.sleep(0.6)

    bot.express("neutral")
    bot.speak("Now go and read example two.")
