# 课堂人脸智能计数系统（PRJ003）

一个面向计算机视觉课程作业的轻量级 Web 应用。用户上传从教室正前方拍摄的
JPG、JPEG 或 PNG 图片后，系统使用 OpenCV YuNet 检测所有可见人脸，在原图
上绘制边界框，显示检测数量，并允许下载标注结果。

> 本系统只做人脸检测和计数，不进行人脸身份识别、学生注册或考勤管理。

## 功能

- 中文 Streamlit 单页面界面，先预览、再点击按钮检测
- 原始图片与检测结果并排预览
- 醒目显示有效人脸框总数
- 可调置信度阈值（默认 `0.50`）
- 大图等比例缩放检测，并把人脸框坐标还原到原图
- 无人脸、损坏图片、异常通道、模型缺失和推理失败提示
- 下载带人脸框的 PNG 结果图
- OpenCV CPU 推理，不要求 NVIDIA 显卡或 CUDA
- 15 MB 文件限制与 1200 万像素限制
- YuNet 模型 SHA-256 校验、云端缺失时的临时缓存回退
- 适配 Streamlit Community Cloud 的 Linux CPU 部署

## 项目结构

```text
graoh2/
├─ app.py                         # Streamlit 页面与交互逻辑
├─ face_detector.py               # YuNet 检测、坐标还原、绘框和编码
├─ requirements.txt               # 运行依赖
├─ requirements-dev.txt           # 验证脚本所需的开发依赖
├─ README.md                       # 安装、运行、原理与答辩说明
├─ DEPLOYMENT.md                   # GitHub 与 Community Cloud 部署教程
├─ .streamlit/config.toml          # 上传限制、主题和无头服务器配置
├─ models/
│  ├─ face_detection_yunet_2023mar.onnx
│  └─ README.md                    # 模型来源说明
├─ scripts/
│  └─ download_model.py            # 模型缺失时的官方源下载脚本
└─ tests/
   ├─ test_face_detector.py        # 解码、绘框和计数一致性单元测试
   └─ manual_verify.py             # 真实 YuNet 三场景验证脚本
```

## Windows 安装与启动

建议使用 Python 3.10～3.12。打开 PowerShell，进入项目目录后执行：

```powershell
cd F:\ckp\graoh2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

浏览器通常会自动打开 `http://localhost:8501`。若 PowerShell 禁止激活脚本，
可以不激活虚拟环境，直接执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

模型已放在 `models/`。若复制项目时模型文件丢失，可联网重新下载：

```powershell
.\.venv\Scripts\python.exe scripts\download_model.py
```

## 使用方法

1. 上传一张不超过 15 MB 的 JPG、JPEG 或 PNG 课堂图片。
2. 查看原始图片预览；需要时展开“检测参数”调整置信度。
3. 点击“开始检测”。
4. 查看大号人脸总数以及原图、标注图的并排对比。
5. 点击“下载标注结果图片”保存 PNG；更换上传图片即可重新检测。

降低置信度可能找回一些远处或模糊的小脸，但也更容易出现误检；提高置信度
会减少误检，也可能漏掉不清晰人脸。课堂图片建议正面、清晰、光线充足。

## 核心计数原理

1. 先检查上传文件大小、真实图片格式和像素总量，再由 `cv2.imdecode` 解码成
   BGR 图片，并统一灰度或带透明通道图片。
2. 若图片最长边超过 2560 像素，先保持宽高比缩小，以控制 CPU 推理耗时。
3. 调用 OpenCV `FaceDetectorYN` 加载 YuNet ONNX 模型进行推理。
4. YuNet 为每个候选人脸输出边界框、5 个关键点和置信度；OpenCV 根据置信度
   阈值过滤候选框，并通过非极大值抑制（NMS）合并重叠框。
5. 如果检测图经过缩放，则将 `(x, y, width, height)` 除以缩放比例，映射回
   原始图片坐标，再把框裁剪到图片边界。
6. 只保留宽、高均有效且置信度达标的框。最终列表长度就是显示的人脸总数，
   绘图也遍历同一个列表，所以检测数字与可见矩形框数量一致。

YuNet 对小尺寸人脸有一定检测能力，但任何通用检测器都不能保证识别每一位
学生。高分辨率大图被缩小时，极小人脸信息仍可能丢失；遮挡、低头、运动模糊、
背光和侧脸也会造成漏检。因此页面明确写的是“检测到的人脸数”，不能直接把它
当作课堂实际人数或考勤结果。

## 预训练模型及来源

项目使用 **OpenCV Zoo 的 YuNet `face_detection_yunet_2023mar.onnx`**：

- OpenCV Zoo 模型页：<https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- YuNet 原始训练代码：<https://github.com/ShiqiYu/libfacedetection.train>
- 论文：*YuNet: A Tiny Millisecond-level Face Detector*

模型由 YuNet 作者训练并由 OpenCV Zoo 分发，本项目只是通过 OpenCV 接口调用
该第三方预训练模型，**不是本项目自行训练的成果**。

项目仓库内附带模型以保证离线运行。模型加载前会核对固定 SHA-256；云端若遗漏
模型，程序会尝试从 OpenCV Zoo 下载到系统临时缓存。上传照片和检测结果只保留
在当前 Streamlit 会话内存中，业务代码不会把它们写入磁盘或数据库。

## 公网部署

完整的 GitHub 上传、Streamlit Community Cloud 部署、验收和 PPT 现场演示步骤见
[DEPLOYMENT.md](DEPLOYMENT.md)。部署时入口文件为 `app.py`，推荐在 Advanced
settings 中选择 Python 3.12。项目使用 `opencv-python-headless`，不需要额外的
Linux 图形包或 `packages.txt`。

## 验证

安装开发依赖后运行单元测试：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

运行真实 YuNet 三场景验证（优先使用本机 scikit-image 标准人物图；若不可用，
会从 OpenCV 官方仓库下载 Lena 测试图）：

```powershell
python tests/manual_verify.py
```

验证包含：纯色无脸图、单人标准图、由三张单人图横向拼接的多人图。输出会写入
`tests/_generated/`，并检查预期数量以及“结果计数 = 绘制框数量”。合成多人图只
用于可重复的软件验证，不代表真实课堂场景的准确率评估。

## 各 Python 文件作用

- `app.py`：配置 Streamlit 页面、中文界面、上传组件、阈值控件、结果展示、
  错误提示和下载按钮；通过缓存避免每次交互都重新加载模型。
- `face_detector.py`：封装图片解码、YuNet CPU 推理、大图缩放与坐标还原、
  有效框过滤、绘制标注和 PNG 编码，是项目的核心算法模块。
- `scripts/download_model.py`：只负责从 OpenCV Zoo 官方地址下载模型，并显示
  文件大小与 SHA-256，方便缺失时恢复。
- `tests/test_face_detector.py`：验证空文件异常、灰度图转换，以及 0/1/4 个框
  时绘图与输出编码行为。
- `tests/manual_verify.py`：使用真实模型验证无脸、单人、多人三种情况。

## 课程答辩建议

可按以下顺序介绍：

1. **需求边界**：目标是上传课堂图片、检测所有可见人脸并计数；不做人脸识别
   和考勤，避免把项目范围扩大成管理系统。
2. **技术选型**：Streamlit 快速构建中文 Web 界面，OpenCV 负责图片处理和模型
   推理，YuNet 模型轻量、适合 CPU。明确模型来自 OpenCV Zoo，不是自行训练。
3. **处理流程**：上传字节 → OpenCV 解码 → 等比例缩放 → YuNet 检测 → NMS →
   坐标还原与裁剪 → 绘框 → 同一框列表计数 → 页面展示和下载。
4. **关键实现**：重点讲缩放比例 `scale`；推理框坐标除以 `scale` 后回到原图，
   解决检测尺寸与显示尺寸不一致的问题。
5. **异常与边界**：模型缺失、坏图、无人脸都不会让页面崩溃；阈值可调；CPU
   后端明确指定。
6. **测试结论**：展示无脸、单人、多人三种验证，并说明计数与绘制框来自同一
   数据列表。最后诚实说明遮挡、低头和远距离小脸仍可能漏检。

## 许可证提示

应用代码为课程项目代码。YuNet 模型及相关权利归其原作者和 OpenCV Zoo 项目，
使用或再分发时请同时遵守相应许可证与署名要求。
