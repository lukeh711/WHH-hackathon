"""Makes `from ohbot_kit import ...` work when running an example directly.

Every example starts with `import _bootstrap  # noqa: F401`. That's one line
instead of an install step, which matters when twenty people are setting up at
once and `pip install -e .` is one more thing to go wrong.

Run examples from the repository root:

    python examples/01_hello_robot.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The ohbot library resolves ohbotData/ (motor calibration, speech database,
# sounds) relative to the working directory. Run from anywhere else and it
# silently creates a second, uncalibrated copy -- so fail loudly instead.
if not os.path.exists(os.path.join(os.getcwd(), "config.yaml")):
    sys.stderr.write(
        f"Run this from the repository root:\n    cd {ROOT}\n    python examples/{os.path.basename(sys.argv[0])}\n"
    )
    sys.exit(1)
