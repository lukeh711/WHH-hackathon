"""Camera capture and the vision-model caption request.

fake_cv2 (tests/conftest.py) stands in for OpenCV, which CI does not install.
"""

from __future__ import annotations

from unittest import mock

import pytest
import requests

from ohbot_kit import llm, vision


class TestCameraOpen:
    def test_discards_exactly_warmup_frames(self, fake_cv2) -> None:
        camera = vision.Camera().open()
        assert camera._capture.read_calls == vision.WARMUP_FRAMES

    def test_raises_camera_error_when_device_will_not_open(self, fake_cv2) -> None:
        fake_cv2.open_succeeds = False
        with pytest.raises(vision.CameraError, match="amera"):
            vision.Camera().open()

    def test_opens_the_requested_index(self, fake_cv2) -> None:
        vision.Camera(index=3).open()
        assert fake_cv2.opened_indexes == [3]


class TestCaptureJpeg:
    def test_resizes_before_encoding_and_returns_bytes(self, fake_cv2) -> None:
        camera = vision.Camera().open()
        jpeg = camera.capture_jpeg()
        assert jpeg == b"jpeg-bytes"
        assert fake_cv2.resized, "resize() was never called"

    def test_returns_none_when_read_fails(self, fake_cv2) -> None:
        camera = vision.Camera().open()
        fake_cv2.frame_present = False
        assert camera.capture_jpeg() is None

    def test_returns_none_when_encode_fails(self, fake_cv2) -> None:
        camera = vision.Camera().open()
        fake_cv2.imencode_succeeds = False
        assert camera.capture_jpeg() is None


class TestCameraClose:
    def test_close_releases_the_device(self, fake_cv2) -> None:
        camera = vision.Camera(index=1).open()
        camera.close()
        assert fake_cv2.released == [1]

    def test_context_manager_opens_and_releases(self, fake_cv2) -> None:
        with vision.Camera(index=2) as camera:
            assert camera._capture is not None
        assert fake_cv2.released == [2]


class TestCameraDescribe:
    def test_no_frame_skips_the_http_call(self, fake_cv2) -> None:
        camera = vision.Camera().open()
        fake_cv2.frame_present = False
        with mock.patch.object(vision.requests, "post") as post:
            assert camera.describe() is None
        post.assert_not_called()

    def test_frame_present_returns_caption(self, fake_cv2) -> None:
        camera = vision.Camera().open()
        response = mock.Mock()
        response.json.return_value = {"response": "A person waving."}
        response.raise_for_status.return_value = None
        with mock.patch.object(vision.requests, "post", return_value=response):
            assert camera.describe() == "A person waving."


class TestDescribe:
    def _response(self, text: str) -> mock.Mock:
        response = mock.Mock()
        response.json.return_value = {"response": text}
        response.raise_for_status.return_value = None
        return response

    def test_posts_image_and_model_without_streaming(self) -> None:
        with mock.patch.object(
            vision.requests, "post", return_value=self._response("A cat.")
        ) as post:
            vision.describe(b"jpeg", model="moondream", host="http://h")
        body = post.call_args.kwargs["json"]
        assert post.call_args.args[0] == "http://h/api/generate"
        assert body["model"] == "moondream"
        assert body["images"] == ["anBlZw=="]
        assert body["stream"] is False

    def test_runs_reply_through_sanitise(self) -> None:
        with mock.patch.object(vision.requests, "post", return_value=self._response("**A cat.**")):
            assert vision.describe(b"jpeg") == "A cat."

    def test_wraps_request_exception_as_ollama_error(self) -> None:
        with (
            mock.patch.object(
                vision.requests, "post", side_effect=requests.RequestException("boom")
            ),
            pytest.raises(llm.OllamaError),
        ):
            vision.describe(b"jpeg")
