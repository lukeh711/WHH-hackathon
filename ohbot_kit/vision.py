"""Webcam capture plus a local Ollama vision model.

Shared by `examples/09_vision.py` and `ohbot_chat.py --camera` so there is one
implementation of the camera-handling quirks below, not two drifting copies.

Not re-exported from ohbot_kit/__init__.py, for the same reason `voice`/`call`
aren't: importing cv2 is pure cost for a chat that never turns the camera on.
Import it directly when you need it.
"""

from __future__ import annotations

import base64
from typing import Any

import requests

from . import llm

DEFAULT_MODEL = "moondream"
TIMEOUT = 180

# Keep this a plain captioning instruction. moondream is a 1.7 GB model and it
# collapses if you ask it to act: measured at temperature 0, 3/3 identical runs
# each time -- "as if you were a friendly robot looking at it" returned a string
# of "!!!!!!", "Mention people if there are any." returned an empty string, and
# any "in one short sentence" phrasing lost the first token of the reply
# ("urns of blue..."). Plain description returns a clean, accurate caption.
#
# The personality belongs to the chat model that receives this caption, which
# turns it into an in-character reaction. Asking the vision model to be a
# character as well is both redundant and what broke it.
PROMPT = "Describe this image, including any people in it."

# macOS hands back a few unexposed (black) frames right after the device opens.
WARMUP_FRAMES = 5

# Downscale before encoding: full-resolution frames cost seconds of inference
# for no gain in what the model notices.
FRAME_SIZE = (512, 384)

_PERMISSION_HINT = (
    "On macOS, grant camera access to the app running this -- System Settings\n"
    "> Privacy & Security > Camera -- then restart it. Otherwise check no other\n"
    "program is holding the camera."
)


class CameraError(RuntimeError):
    """Raised when the camera can't be opened or read."""


def describe(jpeg: bytes, model: str = DEFAULT_MODEL, host: str = llm.HOST) -> str:
    """Ask a local Ollama vision model what's in a JPEG frame.

    Raises llm.OllamaError if the model can't be reached -- the same error the
    chat model raises, so callers can handle both the same way.
    """
    try:
        r = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": PROMPT,
                "images": [base64.b64encode(jpeg).decode()],
                "stream": False,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return llm.sanitise(r.json().get("response", ""))
    except (requests.RequestException, ValueError) as e:
        raise llm.OllamaError(f"Vision request failed: {e}") from e


class Camera:
    """A webcam plus the vision model that describes what it sees.

    Import cv2 lazily inside __init__ rather than at module level, so
    `from ohbot_kit import vision` alone (e.g. for CameraError/DEFAULT_MODEL)
    doesn't require OpenCV to be installed.
    """

    def __init__(self, index: int = 0, model: str = DEFAULT_MODEL, host: str = llm.HOST) -> None:
        import cv2

        self._cv2 = cv2
        self.index = index
        self.model = model
        self.host = host
        self._capture: Any = None

    def open(self) -> Camera:
        capture = self._cv2.VideoCapture(self.index)
        if not capture.isOpened():
            raise CameraError(f"Could not open camera {self.index}.\n\n{_PERMISSION_HINT}")
        # Let the sensor settle before the first look, or the robot opens by
        # describing a black rectangle with total confidence.
        for _ in range(WARMUP_FRAMES):
            capture.read()
        self._capture = capture
        return self

    def capture_jpeg(self) -> bytes | None:
        """Grab one frame and JPEG-encode it. None if the camera gave nothing."""
        assert self._capture is not None, "call open() first"
        ok, frame = self._capture.read()
        if not ok:
            return None
        frame = self._cv2.resize(frame, FRAME_SIZE)
        ok, buf = self._cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return buf.tobytes()

    def describe(self) -> str | None:
        """Capture a frame and describe it. None means no frame was available --
        distinct from an empty string, which means the model saw nothing worth
        describing (e.g. a covered lens)."""
        jpeg = self.capture_jpeg()
        if jpeg is None:
            return None
        return describe(jpeg, self.model, self.host)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Camera:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()
