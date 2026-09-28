"""Personas: one robot, different characters.

    python examples/05_personas.py
    python examples/05_personas.py pirate

A persona in config.yaml pairs a system prompt with a voice. Add your own
there -- it needs nothing but a name, a `voice:` and a `system_prompt:`.

This is the cheapest way to make your project feel distinct: same code, and
the robot becomes a pirate, a teacher, or whatever your team invents.
"""

import sys

import _bootstrap  # noqa: F401

from ohbot_kit import Ohbot, setup
from ohbot_kit import config as config_mod

QUESTION = "Who are you, and what do you like doing?"

cfg = config_mod.load(warn=False)
wanted = sys.argv[1:] or sorted(cfg.section("personas"))

for name in wanted:
    print(f"\n=== {name} ===")
    # setup() re-reads config and installs that persona's voice.
    _cfg, convo, robot_kwargs = setup(persona=name)

    # Idle motion off so the personas are easier to compare.
    robot_kwargs["idle"] = False
    with Ohbot(**robot_kwargs) as bot:
        for sentence in convo.stream_sentences(QUESTION):
            print(sentence)
            bot.speak(sentence)
