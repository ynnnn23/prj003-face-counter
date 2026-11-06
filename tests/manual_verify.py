"""使用真实 YuNet 模型验证无脸、单人和多人三类场景。

脚本会从 OpenCV 官方仓库下载经典 Lena 测试图，并把同一张人脸拼成多人图。
生成文件仅用于本地验证，不作为系统业务数据。
"""

from __future__ import annotations

import urllib.request
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np

# 允许直接执行 ``python tests/manual_verify.py``。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_detector import DEFAULT_MODEL_PATH, YuNetFaceDetector, draw_face_boxes


LENA_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "_generated"


def load_test_face() -> np.ndarray:
    """优先使用本机 scikit-image 测试图，否则下载 OpenCV Lena。"""

    skimage_spec = importlib.util.find_spec("skimage")
    if skimage_spec and skimage_spec.origin:
        astronaut_path = Path(skimage_spec.origin).parent / "data" / "astronaut.png"
        if astronaut_path.is_file():
            image = cv2.imread(str(astronaut_path), cv2.IMREAD_COLOR)
            if image is not None:
                print(f"使用本机标准测试图：{astronaut_path}")
                return image

    request = urllib.request.Request(LENA_URL, headers={"User-Agent": "PRJ003-test/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = np.frombuffer(response.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Lena 测试图下载后无法解码。")
    return image


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = YuNetFaceDetector(DEFAULT_MODEL_PATH)
    face_image = load_test_face()

    blank = np.full_like(face_image, 245)
    multi = np.hstack([face_image, face_image, face_image])
    cases = {
        "无脸图片": (blank, 0),
        "单人图片": (face_image, 1),
        "多人图片": (multi, 3),
    }

    failed = False
    for name, (image, expected) in cases.items():
        result = detector.detect(image, confidence_threshold=0.5)
        boxes_drawn = len(result.faces)
        consistent = result.count == boxes_drawn
        passed = result.count == expected and consistent
        print(
            f"{name}: 期望={expected}, 检测={result.count}, "
            f"绘制框={boxes_drawn}, {'通过' if passed else '未通过'}"
        )
        cv2.imwrite(str(OUTPUT_DIR / f"{name}.png"), draw_face_boxes(image, result.faces))
        failed = failed or not passed

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
