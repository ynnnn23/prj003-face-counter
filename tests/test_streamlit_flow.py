"""验证上传、点击检测、人数显示与 PNG 下载的完整页面流程。"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_blank_image_full_web_flow() -> None:
    blank = np.full((240, 320, 3), 245, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", blank)
    assert ok

    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception

    uploader = app.get("file_uploader")[0]
    uploader.upload("no-face.png", encoded.tobytes(), "image/png").run()
    assert not app.exception
    assert any("原始图片预览" in str(item.value) for item in app.markdown)

    app.button(key="start_detection").click().run(timeout=30)
    assert not app.exception
    assert any("检测到" in str(item.value) and "0" in str(item.value) for item in app.markdown)

    downloads = app.get("download_button")
    assert len(downloads) == 1
    # AppTest 把下载内容放入内部媒体管理器，PNG 类型会反映在模拟下载 URL 中。
    assert downloads[0].proto.url.endswith(".png")

    # 更换图片后必须清理上一张结果，等待用户重新点击检测。
    second = np.full((240, 320, 3), 210, dtype=np.uint8)
    ok, second_encoded = cv2.imencode(".png", second)
    assert ok
    app.get("file_uploader")[0].upload(
        "another.png", second_encoded.tobytes(), "image/png"
    ).run()
    assert not app.exception
    assert len(app.get("download_button")) == 0
    assert app.button(key="start_detection").label == "\u5f00\u59cb\u68c0\u6d4b"
