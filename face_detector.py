"""基于 OpenCV YuNet 的人脸检测模块。

本模块只负责人脸检测、坐标还原和结果绘制，不包含身份识别功能。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from threading import Lock
import tempfile
import time
from typing import Sequence
import urllib.error
import urllib.request

import cv2
import numpy as np


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)
MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    f"face_detection_yunet/{MODEL_FILENAME}"
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODEL_MIN_BYTES = 100_000


class FaceDetectorError(RuntimeError):
    """人脸检测过程中可向界面展示的基础异常。"""


class ModelFileNotFoundError(FaceDetectorError):
    """YuNet 模型文件不存在。"""


class InvalidImageError(FaceDetectorError):
    """上传内容无法被解析为有效图片。"""


def file_sha256(path: str | Path) -> str:
    """分块计算文件 SHA-256，避免一次性读取大文件。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_yunet_model(path: str | Path) -> Path:
    """验证 YuNet 文件存在、大小合理且哈希与官方版本一致。"""

    model_path = Path(path)
    if not model_path.is_file():
        raise ModelFileNotFoundError(f"未找到 YuNet 模型文件：{model_path}")
    if model_path.stat().st_size < MODEL_MIN_BYTES:
        raise FaceDetectorError(f"YuNet 模型文件不完整：{model_path}")

    actual_hash = file_sha256(model_path)
    if actual_hash.lower() != MODEL_SHA256:
        raise FaceDetectorError(
            "YuNet 模型完整性校验失败。"
            f"期望 SHA-256：{MODEL_SHA256}，实际：{actual_hash}。"
        )
    return model_path


def download_yunet_model(
    destination: str | Path,
    *,
    retries: int = 3,
    timeout: int = 60,
) -> Path:
    """从 OpenCV Zoo 下载并原子写入 YuNet 模型。

    每次下载后都执行 SHA-256 校验；只有校验成功才替换目标文件，避免网络中断
    留下损坏模型。
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    last_error: Exception | None = None

    for attempt in range(1, max(1, retries) + 1):
        try:
            request = urllib.request.Request(
                MODEL_URL,
                headers={"User-Agent": "PRJ003-classroom-face-counter/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read()

            temporary_path.write_bytes(content)
            verify_yunet_model(temporary_path)
            os.replace(temporary_path, destination)
            return verify_yunet_model(destination)
        except (OSError, urllib.error.URLError, FaceDetectorError) as exc:
            last_error = exc
            temporary_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(attempt, 2))

    raise FaceDetectorError(
        "无法从 OpenCV Zoo 获取 YuNet 模型。请检查网络，或在联网电脑上运行 "
        "python scripts/download_model.py 后将 models 目录一并部署。"
        f"最后一次错误：{last_error}"
    ) from last_error


def resolve_yunet_model() -> Path:
    """优先使用项目内模型，缺失时下载到系统临时缓存目录。

    正常的本地离线运行和 Community Cloud 部署都会直接使用仓库中的模型，不会
    发起网络请求。临时缓存只作为模型遗漏时的容错方案。
    """

    try:
        return verify_yunet_model(DEFAULT_MODEL_PATH)
    except FaceDetectorError:
        cache_path = Path(tempfile.gettempdir()) / "prj003_yunet" / MODEL_FILENAME
        try:
            return verify_yunet_model(cache_path)
        except FaceDetectorError:
            return download_yunet_model(cache_path)


@dataclass(frozen=True)
class FaceBox:
    """一张人脸在原始图片坐标系中的边界框。"""

    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass(frozen=True)
class DetectionResult:
    """一次检测的结构化结果。"""

    faces: tuple[FaceBox, ...]
    original_size: tuple[int, int]  # (宽, 高)
    detection_size: tuple[int, int]  # (宽, 高)
    scale: float

    @property
    def count(self) -> int:
        """有效人脸框数量，也是界面显示的人脸数。"""

        return len(self.faces)


def decode_image(image_bytes: bytes) -> np.ndarray:
    """将上传文件字节解码为 OpenCV BGR 图片。

    OpenCV 的 ``imdecode`` 能处理包含中文路径时常见的读图问题，也便于直接
    读取 Streamlit 上传组件返回的内存字节。
    """

    if not image_bytes:
        raise InvalidImageError("上传文件为空，请重新选择图片。")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise InvalidImageError("图片无法解码，文件可能已损坏或格式不受支持。")

    # 统一为 YuNet 所需的 3 通道 BGR 格式。
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError("图片通道格式异常，请上传 JPG、JPEG 或 PNG 图片。")

    return image


class YuNetFaceDetector:
    """使用 OpenCV ``FaceDetectorYN`` 调用 YuNet 预训练模型。"""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_input_side: int = 2560,
    ) -> None:
        self.model_path = Path(model_path)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.max_input_side = int(max_input_side)
        self._lock = Lock()

        if not self.model_path.is_file():
            raise ModelFileNotFoundError(
                f"未找到 YuNet 模型文件：{self.model_path}。"
                "请按照 README.md 的说明下载模型。"
            )
        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise FaceDetectorError(
                "当前 OpenCV 版本不支持 FaceDetectorYN，请安装 requirements.txt 中的版本。"
            )
        if self.max_input_side < 320:
            raise ValueError("max_input_side 不能小于 320。")

        try:
            # 明确指定 OpenCV 后端与 CPU 目标，不依赖 CUDA。
            self._detector = cv2.FaceDetectorYN_create(
                str(self.model_path),
                "",
                (320, 320),
                0.5,
                self.nms_threshold,
                self.top_k,
                cv2.dnn.DNN_BACKEND_OPENCV,
                cv2.dnn.DNN_TARGET_CPU,
            )
        except cv2.error as exc:
            raise FaceDetectorError(f"YuNet 模型加载失败：{exc}") from exc

    def detect(
        self, image: np.ndarray, confidence_threshold: float = 0.5
    ) -> DetectionResult:
        """检测图片中的人脸，并返回原图坐标系下的有效边界框。

        当图片过大时先等比例缩小用于推理，再用缩放比例把检测框坐标还原到
        原图。框会被裁剪到原图边界，宽或高无效的框不会参与计数。
        """

        self._validate_image(image)
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("置信度阈值必须位于 (0, 1]。")

        original_height, original_width = image.shape[:2]
        scale = min(1.0, self.max_input_side / max(original_width, original_height))

        if scale < 1.0:
            detection_width = max(1, round(original_width * scale))
            detection_height = max(1, round(original_height * scale))
            detection_image = cv2.resize(
                image,
                (detection_width, detection_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            detection_image = image
            detection_width, detection_height = original_width, original_height

        try:
            # FaceDetectorYN 内部保存输入尺寸和阈值，因此用锁保证并发会话安全。
            with self._lock:
                self._detector.setInputSize((detection_width, detection_height))
                self._detector.setScoreThreshold(float(confidence_threshold))
                _, raw_faces = self._detector.detect(detection_image)
        except cv2.error as exc:
            raise FaceDetectorError(f"OpenCV 人脸检测失败：{exc}") from exc

        faces: list[FaceBox] = []
        if raw_faces is not None:
            for row in np.asarray(raw_faces):
                if row.size < 15:
                    continue

                x, y, width, height = (float(value) / scale for value in row[:4])
                score = float(row[-1])

                # 使用左上角和右下角裁剪，避免框超出原图。
                x1 = int(np.clip(round(x), 0, original_width - 1))
                y1 = int(np.clip(round(y), 0, original_height - 1))
                x2 = int(np.clip(round(x + width), x1 + 1, original_width))
                y2 = int(np.clip(round(y + height), y1 + 1, original_height))
                box_width, box_height = x2 - x1, y2 - y1

                if box_width > 0 and box_height > 0 and score >= confidence_threshold:
                    faces.append(FaceBox(x1, y1, box_width, box_height, score))

        return DetectionResult(
            faces=tuple(faces),
            original_size=(original_width, original_height),
            detection_size=(detection_width, detection_height),
            scale=scale,
        )

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise InvalidImageError("输入图片为空或类型无效。")
        if image.ndim != 3 or image.shape[2] != 3:
            raise InvalidImageError("检测器要求 3 通道 BGR 图片。")
        if image.dtype != np.uint8:
            raise InvalidImageError("检测器要求 uint8 类型图片。")


def draw_face_boxes(
    image: np.ndarray,
    faces: Sequence[FaceBox],
    *,
    show_confidence: bool = True,
) -> np.ndarray:
    """在图片副本上绘制人脸框、序号和置信度。"""

    annotated = image.copy()
    image_height, image_width = annotated.shape[:2]
    line_width = max(2, round(min(image_width, image_height) / 450))
    font_scale = max(0.5, min(1.2, min(image_width, image_height) / 900))

    for index, face in enumerate(faces, start=1):
        top_left = (face.x, face.y)
        bottom_right = (face.x + face.width, face.y + face.height)
        cv2.rectangle(annotated, top_left, bottom_right, (38, 208, 124), line_width)

        label = f"#{index}"
        if show_confidence:
            label += f"  {face.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        label_y1 = max(0, face.y - text_height - baseline - 8)
        label_y2 = label_y1 + text_height + baseline + 8
        label_x2 = min(image_width, face.x + text_width + 10)
        cv2.rectangle(
            annotated, (face.x, label_y1), (label_x2, label_y2), (38, 208, 124), -1
        )
        cv2.putText(
            annotated,
            label,
            (face.x + 5, label_y2 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_width,
            cv2.LINE_AA,
        )

    return annotated


def encode_png(image: np.ndarray) -> bytes:
    """把 BGR 图片编码为可下载的 PNG 字节。"""

    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise FaceDetectorError("检测结果图片编码失败。")
    return buffer.tobytes()
