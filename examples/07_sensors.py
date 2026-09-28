"""React to the world: reading Ohbot's sensor.

    python examples/07_sensors.py

Ohbot has sensor pins 0-6, each returning a float from 0 to 10. With the
distance sensor fitted, this makes the robot notice someone approaching --
which is a far better demo opening than a keypress.

Use bot.read_sensor(), not ohbot.readSensor(): the library version calls
ser.flushInput() with no guard, so it raises AttributeError on None when no
robot is attached, and it does a write-then-read that another thread's gesture
could land in the middle of. The wrapper holds the serial lock across the whole
transaction and returns 0.0 when there's no robot.
"""

import time

import _bootstrap  # noqa: F401

from ohbot_kit import Ohbot, setup

SENSOR = 0  # pin number
NEAR = 5.0  # tune this: print the readings first and pick a threshold
POLL = 0.2

cfg, _convo, robot_kwargs = setup()

print(f"Reading sensor {SENSOR} -- wave your hand in front of it. Ctrl-C to stop.\n")

with Ohbot(**robot_kwargs) as bot:
    was_near = False
    try:
        while True:
            value = bot.read_sensor(SENSOR)
            is_near = value > NEAR
            print(
                "\r  sensor {} = {:5.2f}  {}".format(SENSOR, value, "NEAR" if is_near else "    "),
                end="",
                flush=True,
            )

            # Only react on the transition, or it greets you continuously.
            if is_near and not was_near:
                print()
                bot.look_at("user")
                bot.speak("Oh, hello there!", emotion="surprised", gesture="perk_up")
            elif was_near and not is_near:
                print()
                bot.speak("Goodbye then.", emotion="sad", gesture="slow_nod")

            was_near = is_near
            time.sleep(POLL)
    except KeyboardInterrupt:
        print()
