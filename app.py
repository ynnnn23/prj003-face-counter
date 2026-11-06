"""PRJ003：课堂人脸智能计数系统 Streamlit 主程序。"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import warnings

import cv2
from PIL import Image, UnidentifiedImageError
import streamlit as st

from face_detector import (
    FaceDetectorError,
    InvalidImageError,
    ModelFileNotFoundError,
    YuNetFaceDetector,
    decode_image,
    draw_face_boxes,
    encode_png,
    resolve_yunet_model,
)


MAX_UPLOAD_MB = 15
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000


st.set_page_config(
    page_title="课堂人脸智能计数系统",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_styles() -> None:
    """注入响应式样式，使课堂演示页面简洁、现代。"""

    st.markdown(
        """
        <style>
        :root {
            --primary: #174ea6;
            --primary-light: #eaf1fc;
            --accent: #16856b;
            --ink: #15233c;
            --muted: #657188;
            --line: #dfe6f1;
            --surface: #ffffff;
        }
        .stApp { background: #f4f7fb; color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(244, 247, 251, .88); }
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.2rem 2.35rem;
            border-radius: 24px;
            color: white;
            background: linear-gradient(125deg, #123b7c 0%, #2364c4 57%, #16856b 115%);
            box-shadow: 0 18px 42px rgba(25, 64, 127, .18);
            margin-bottom: 1.35rem;
        }
        .hero::after {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            border-radius: 50%;
            right: -70px;
            top: -110px;
            background: rgba(255, 255, 255, .10);
        }
        .project-tag {
            display: inline-block;
            padding: .3rem .7rem;
            margin-bottom: .8rem;
            border: 1px solid rgba(255, 255, 255, .35);
            border-radius: 999px;
            font-size: .82rem;
            font-weight: 700;
            letter-spacing: .08em;
        }
        .hero h1 { margin: 0 0 .5rem; font-size: clamp(1.8rem, 4vw, 2.55rem); }
        .hero p { margin: 0; max-width: 720px; opacity: .9; font-size: 1.02rem; }
        .section-title { margin: 1.65rem 0 .3rem; font-size: 1.25rem; font-weight: 750; }
        .section-note { color: var(--muted); margin-bottom: .8rem; }
        .step-card {
            min-height: 104px;
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: var(--surface);
            border: 1px solid var(--line);
            box-shadow: 0 6px 20px rgba(29, 55, 91, .055);
        }
        .step-number { color: var(--primary); font-size: .78rem; font-weight: 800; }
        .step-title { margin-top: .25rem; color: var(--ink); font-size: 1rem; font-weight: 750; }
        .step-text { margin-top: .15rem; color: var(--muted); font-size: .88rem; }
        .count-card {
            text-align: center;
            padding: 1.7rem 1rem;
            border-radius: 20px;
            background: linear-gradient(145deg, #ffffff, #eef5ff);
            border: 1px solid #cadbf4;
            box-shadow: 0 12px 28px rgba(32, 79, 145, .11);
        }
        .count-sentence { color: var(--ink); font-size: 1.1rem; font-weight: 700; }
        .count-number {
            display: inline-block;
            margin: 0 .32rem;
            color: var(--primary);
            font-size: clamp(3rem, 8vw, 4.8rem);
            font-weight: 850;
            line-height: 1;
        }
        .count-detail { margin-top: .65rem; color: var(--accent); font-size: .9rem; font-weight: 650; }
        .privacy-note {
            margin-top: 1rem;
            padding: .8rem 1rem;
            color: #526076;
            background: #edf2f8;
            border-radius: 12px;
            font-size: .88rem;
        }
        [data-testid="stFileUploaderDropzone"] {
            min-height: 126px;
            background: white;
            border: 1.5px dashed #789ad0;
            border-radius: 18px;
        }
        [data-testid="stImage"] img {
            border-radius: 16px;
            border: 1px solid var(--line);
            box-shadow: 0 7px 20px rgba(29, 55, 91, .07);
        }
        [data-testid="stButton"] button[kind="primary"] {
            min-height: 3rem;
            border-radius: 12px;
            font-weight: 750;
        }
        [data-testid="stDownloadButton"] button {
            min-height: 3rem;
            border-radius: 12px;
            font-weight: 700;
        }
        @media (max-width: 768px) {
            .block-container { padding: 1rem .9rem 2rem; }
            .hero { padding: 1.55rem 1.3rem; border-radius: 18px; }
            .hero p { font-size: .93rem; }
            .step-card { min-height: auto; margin-bottom: .35rem; }
            .count-card { padding: 1.25rem .75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_detector() -> YuNetFaceDetector:
    """解析并缓存模型；正常部署直接使用仓库内的 ONNX 文件。"""

    model_path = resolve_yunet_model()
    return YuNetFaceDetector(model_path)


def bgr_to_rgb(image):
    """将 OpenCV 的 BGR 顺序转换为 Streamlit 使用的 RGB。"""

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def validate_uploaded_image(image_bytes: bytes) -> tuple[int, int]:
    """在 OpenCV 解码前检查文件大小、真实格式和像素总量。"""

    if not image_bytes:
        raise InvalidImageError("上传文件为空，请重新选择图片。")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise InvalidImageError(f"图片文件不能超过 {MAX_UPLOAD_MB} MB。")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as probe:
                width, height = probe.size
                image_format = (probe.format or "").upper()
                if image_format not in {"JPEG", "PNG"}:
                    raise InvalidImageError("文件内容不是有效的 JPG、JPEG 或 PNG 图片。")
                if width <= 0 or height <= 0:
                    raise InvalidImageError("图片尺寸无效。")
                if width * height > MAX_IMAGE_PIXELS:
                    raise InvalidImageError(
                        "图片分辨率过大；请使用不超过 1200 万像素的图片。"
                    )
                probe.verify()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("图片损坏或格式不受支持，请重新选择图片。") from exc

    return width, height


def safe_download_stem(filename: str) -> str:
    """生成仅用于浏览器下载的安全文件名，不在服务器持久化。"""

    stem = Path(filename).stem[:60]
    cleaned = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", stem).strip("_")
    return cleaned or "课堂图片"


def reset_result_for_new_upload(upload_digest: str) -> None:
    """选择新图片时清理上一张图片的内存结果。"""

    if st.session_state.get("upload_digest") != upload_digest:
        st.session_state.upload_digest = upload_digest
        st.session_state.pop("detection_bundle", None)


def render_steps() -> None:
    """展示简洁的三步操作说明。"""

    columns = st.columns(3)
    steps = (
        ("步骤 01", "上传课堂图片", "支持 JPG、JPEG、PNG，最大 15 MB"),
        ("步骤 02", "点击开始检测", "YuNet 在 CPU 上自动定位人脸"),
        ("步骤 03", "查看并下载", "核对人脸数量并保存标注结果"),
    )
    for column, (number, title, text) in zip(columns, steps):
        with column:
            st.markdown(
                f"""
                <div class="step-card">
                  <div class="step-number">{number}</div>
                  <div class="step-title">{title}</div>
                  <div class="step-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_result(image, bundle: dict, uploaded_name: str) -> None:
    """展示人数、原图、标注图与下载按钮。"""

    result = bundle["result"]
    annotated = bundle["annotated"]

    st.markdown('<div class="section-title">检测结果</div>', unsafe_allow_html=True)
    count_column, info_column = st.columns([1.1, 2], vertical_alignment="center")
    with count_column:
        st.markdown(
            f"""
            <div class="count-card">
              <div class="count-sentence">检测到 <span class="count-number">{result.count}</span> 张人脸</div>
              <div class="count-detail">共绘制 {len(result.faces)} 个有效人脸框</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_column:
        if result.count == 0:
            st.warning(
                "未检测到人脸。可适当降低置信度，或换用更清晰、光线更充足的正面图片。"
            )
        else:
            st.success("检测完成，标注结果已生成。")
        st.caption(
            f"检测阈值：{bundle['confidence']:.2f}　|　"
            f"原图：{result.original_size[0]} × {result.original_size[1]}　|　"
            f"推理图：{result.detection_size[0]} × {result.detection_size[1]}"
        )
        if result.scale < 1.0:
            st.caption("大图已等比例缩小用于推理，所有检测框均已映射回原图坐标。")

    original_column, result_column = st.columns(2)
    with original_column:
        st.markdown("#### 原始图片")
        st.image(bgr_to_rgb(image), width="stretch")
    with result_column:
        st.markdown("#### 人脸检测标注")
        st.image(bgr_to_rgb(annotated), width="stretch")

    st.download_button(
        "下载标注结果图片",
        data=bundle["download_bytes"],
        file_name=f"{safe_download_stem(uploaded_name)}_人脸检测结果.png",
        mime="image/png",
        width="stretch",
        key="download_result",
    )


def main() -> None:
    apply_styles()

    st.markdown(
        """
        <section class="hero">
          <div class="project-tag">PRJ003 · 计算机视觉课程项目</div>
          <h1>课堂人脸智能计数系统</h1>
          <p>上传一张课堂照片，使用 OpenCV YuNet 自动检测、标注并统计画面中的可见人脸。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_steps()

    st.markdown('<div class="section-title">上传图片</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">请选择从教室正前方拍摄、画面清晰的课堂照片。</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "上传图片",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help=f"支持 JPG、JPEG、PNG；文件不超过 {MAX_UPLOAD_MB} MB。",
        label_visibility="collapsed",
        key="classroom_image",
    )

    if uploaded_file is None:
        st.info("上传图片后会先显示原图预览；确认无误后点击“开始检测”。")
        st.markdown(
            """
            <div class="privacy-note">
              隐私说明：图片仅在当前会话内存中用于检测，本项目不会主动写入数据库或保存上传照片。
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        image_bytes = uploaded_file.getvalue()
        width, height = validate_uploaded_image(image_bytes)
        upload_digest = sha256(image_bytes).hexdigest()
        reset_result_for_new_upload(upload_digest)
        image = decode_image(image_bytes)

        st.caption(
            f"已选择：{uploaded_file.name}　|　{width} × {height}　|　"
            f"{len(image_bytes) / 1024 / 1024:.2f} MB"
        )

        with st.expander("检测参数", expanded=False):
            confidence = st.slider(
                "人脸检测置信度阈值",
                min_value=0.20,
                max_value=0.95,
                value=0.50,
                step=0.05,
                help="降低阈值可能找到更多远处小脸，但也可能增加误检。",
                key="confidence_threshold",
            )
            st.caption("推荐先使用 0.50；远处人脸漏检时，可尝试降低到 0.35～0.45。")

        bundle = st.session_state.get("detection_bundle")
        if bundle is None:
            st.markdown("#### 原始图片预览")
            st.image(bgr_to_rgb(image), width="stretch")

        button_text = "开始检测" if bundle is None else "使用当前参数重新检测"
        if st.button(
            button_text,
            type="primary",
            width="stretch",
            key="start_detection",
        ):
            with st.spinner("正在使用 YuNet 检测人脸，请稍候……"):
                detector = load_detector()
                result = detector.detect(image, confidence_threshold=confidence)
                annotated = draw_face_boxes(image, result.faces)
                st.session_state.detection_bundle = {
                    "result": result,
                    "annotated": annotated,
                    "download_bytes": encode_png(annotated),
                    "confidence": confidence,
                    "upload_digest": upload_digest,
                }
                bundle = st.session_state.detection_bundle

        if bundle is not None and bundle.get("upload_digest") == upload_digest:
            if bundle["confidence"] != confidence:
                st.info("置信度参数已改变；点击“使用当前参数重新检测”可更新结果。")
            render_result(image, bundle, uploaded_file.name)

        st.markdown(
            """
            <div class="privacy-note">
              图片仅在当前会话中处理，不写入本地文件或数据库。检测到的人脸数不等同于课堂实际人数；
              遮挡、低头、侧脸、模糊和光线不足都可能造成漏检。
            </div>
            """,
            unsafe_allow_html=True,
        )

    except (InvalidImageError, ModelFileNotFoundError, FaceDetectorError) as exc:
        st.error(str(exc))
        if isinstance(exc, (ModelFileNotFoundError, FaceDetectorError)):
            st.caption("本地离线使用前，可先执行以下命令准备模型：")
            st.code("python scripts/download_model.py", language="powershell")
    except Exception as exc:  # 防止未知错误让页面直接崩溃，同时保留调试入口。
        st.error("处理图片时发生未预期错误，请检查图片和运行环境。")
        with st.expander("查看错误详情"):
            st.exception(exc)


if __name__ == "__main__":
    main()
