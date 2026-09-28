"""Give the robot eyes: webcam plus a local vision model.

    ollama pull moondream          # ~1.7 GB, small and quick
    python examples/09_vision.py

    python examples/09_vision.py --model qwen3.6:35b   # smarter, much slower

OpenCV is already in requirements.txt; this is the only example that uses it.

Grabs one frame from the webcam, asks a vision model what it sees, and the
robot reacts to it out loud.

WHY A SMALL MODEL: the loop only feels alive if it answers in a couple of
seconds. Measured on the same frame, moondream took 1.1-1.2s and qwen3.6:35b
took 21s. qwen sees more detail, but 21s of a motionless robot reads as a
crash -- fine for a single party trick, poor for interaction.

The very first look is slower (~10s) while Ollama loads the vision model;
check_setup.py warms the chat model but not this one. Every look after that is
the figure above.

CAMERA PERMISSION: macOS may deny this silently rather than asking. If OpenCV
says "not authorized to capture video", grant camera access to the app you run
Python from -- VS Code, if that is where your terminal lives, not Terminal.app.
`python tools/check_setup.py` reports the camera and the vision model.

Before you rewrite the captioning prompt, read the comment above it in
ohbot_kit/vision.py. Asking a 1.7 GB model to act in character instead of
describe makes it emit "!!!!!!".
"""

import argparse
import sys
import time

import _bootstrap  # noqa: F401

from ohbot_kit import Ohbot, expression, llm, setup, vision

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--model", default="moondream", help="an Ollama model with vision")
p.add_argument("--interval", type=float, default=8.0, help="seconds between looks")
p.add_argument("--camera", type=int, default=0)
args = p.parse_args()

# Messages of history to keep -- two exchanges. See the trim in the loop below.
HISTORY = 4

cfg, convo, robot_kwargs = setup()
host = cfg.get("llm.host", llm.HOST)

try:
    llm.check_model(args.model, host)
except llm.OllamaError as e:
    sys.exit(f"{e}\n\nInstall it with:  ollama pull {args.model}")

try:
    camera = vision.Camera(args.camera, args.model, host).open()
except vision.CameraError as e:
    sys.exit(str(e))

print(f"Looking every {args.interval:.0f}s. Ctrl-C to stop.\n")

try:
    with Ohbot(**robot_kwargs) as bot:
        bot.speak("Let me have a look around.", emotion="curious", gesture="tilt")
        while True:
            bot.set_state("thinking")
            bot.express("thinking")
            t0 = time.time()

            # Every skipped turn still waits out the interval. A bare `continue`
            # here would spin at full speed against a camera or a model that has
            # stopped answering, which is exactly when you least want that.
            try:
                description = camera.describe()
            except llm.OllamaError as e:
                print(f"[vision] {e}", file=sys.stderr)
                time.sleep(args.interval)
                continue
            if description is None:
                print("[vision] no frame from the camera", file=sys.stderr)
                time.sleep(args.interval)
                continue
            if not description:
                # Distinct from the above on purpose: a small vision model does
                # sometimes answer with punctuation or emoji that sanitise()
                # strips to nothing. That is the model, not the camera.
                print("[vision] nothing describable in that frame", file=sys.stderr)
                time.sleep(args.interval)
                continue
            print(f"[{time.time() - t0:.1f}s] {description}")

            # Let the chat model react in character, with a face and a gesture,
            # rather than the robot flatly reading the caption aloud.
            try:
                action = convo.respond_with_action(
                    f"You just looked around and saw: {description}. React briefly.",
                    expression.EMOTIONS,
                    expression.GESTURE_NAMES,
                )
            except llm.OllamaError as e:
                print(f"[llm] {e}", file=sys.stderr)
                time.sleep(args.interval)
                continue

            # Unlike the chat examples this loop is unattended, and every reply
            # appends two messages to the history. Left alone it grows until
            # latency climbs and the context overflows. Two exchanges is enough
            # to stop the robot repeating itself at an unchanged scene.
            del convo.messages[:-HISTORY]

            bot.speak(action["say"], emotion=action["emotion"], gesture=action["gesture"])
            time.sleep(args.interval)
except KeyboardInterrupt:
    print()
finally:
    camera.close()
