"""从 OpenCV Zoo 官方仓库下载 YuNet 预训练模型。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_detector import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    MODEL_URL,
    download_yunet_model,
    file_sha256,
)


def main() -> int:
    print(f"正在从 OpenCV Zoo 下载 YuNet：\n{MODEL_URL}")
    try:
        model_path = download_yunet_model(DEFAULT_MODEL_PATH)
    except Exception as exc:
        print(f"模型下载失败：{exc}", file=sys.stderr)
        return 1

    print(f"下载完成：{model_path}")
    print(f"文件大小：{model_path.stat().st_size:,} 字节")
    print(f"SHA-256：{file_sha256(model_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
