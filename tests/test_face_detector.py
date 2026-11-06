"""检测模块的轻量单元测试，不依赖真实模型推理。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import face_detector
from face_detector import (
    DEFAULT_MODEL_PATH,
    FaceBox,
    InvalidImageError,
    MODEL_SHA256,
    YuNetFaceDetector,
    decode_image,
    draw_face_boxes,
    encode_png,
    file_sha256,
    verify_yunet_model,
)


def test_decode_empty_file_raises_readable_error() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"")


def test_bundled_yunet_model_integrity() -> None:
    assert verify_yunet_model(DEFAULT_MODEL_PATH) == DEFAULT_MODEL_PATH
    assert file_sha256(DEFAULT_MODEL_PATH) == MODEL_SHA256


def test_decode_grayscale_image_to_bgr() -> None:
    grayscale = np.zeros((30, 40), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", grayscale)
    assert ok
    decoded = decode_image(encoded.tobytes())
    assert decoded.shape == (30, 40, 3)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_drawn_box_count_matches_input_face_count(count: int) -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    faces = tuple(
        FaceBox(x=10 + i * 55, y=40, width=45, height=55, confidence=0.9)
        for i in range(count)
    )
    annotated = draw_face_boxes(image, faces)

    assert len(faces) == count
    assert annotated.shape == image.shape
    assert encode_png(annotated).startswith(b"\x89PNG")


def test_scaled_detection_maps_box_back_to_original(monkeypatch, tmp_path) -> None:
    """验证缩小推理后的边界框会正确还原到原图坐标。"""

    class DummyDetector:
        def __init__(self) -> None:
            self.input_size = None

        def setInputSize(self, size) -> None:  # OpenCV 接口使用驼峰命名
            self.input_size = size

        def setScoreThreshold(self, threshold) -> None:
            self.threshold = threshold

        def detect(self, image):
            # 检测图上的框为 (100, 50, 200, 100)，其余 10 个数模拟关键点与分数。
            row = np.array(
                [[100, 50, 200, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9]],
                dtype=np.float32,
            )
            return 1, row

    dummy = DummyDetector()
    monkeypatch.setattr(face_detector.cv2, "FaceDetectorYN_create", lambda *args: dummy)
    model_path = tmp_path / "mock.onnx"
    model_path.write_bytes(b"mock")

    detector = YuNetFaceDetector(model_path, max_input_side=1000)
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)
    result = detector.detect(image, confidence_threshold=0.5)

    assert result.scale == 0.5
    assert result.detection_size == (1000, 500)
    assert dummy.input_size == (1000, 500)
    assert result.faces == (FaceBox(200, 100, 400, 200, pytest.approx(0.9)),)
