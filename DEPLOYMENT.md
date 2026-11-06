# PRJ003 部署指南

本项目可以直接部署到 Streamlit Community Cloud。云端运行环境使用 Linux CPU，
项目已使用 `opencv-python-headless`，不依赖桌面图形库、CUDA 或 Windows 路径。

## 部署前检查

需要提交到 GitHub 的核心文件：

```text
app.py
face_detector.py
requirements.txt
.streamlit/config.toml
models/face_detection_yunet_2023mar.onnx
models/README.md
```

不要提交：

- `.venv/`、`venv/`、`__pycache__/` 和 `.pytest_cache/`
- `.env`、`.streamlit/secrets.toml` 或任何账号密钥
- 测试生成目录 `tests/_generated/`
- 真实学生照片、课堂原始照片或其他个人数据

YuNet 模型只有约 227 KB，建议作为普通 Git 文件提交，不需要 Git LFS。程序会先
验证模型大小和 SHA-256。若部署时模型确实缺失，则尝试从 OpenCV Zoo 下载到系统
临时缓存目录；下载失败时页面会显示可操作的错误提示。

## 上传到自己的 GitHub 仓库

先在 GitHub 网页上创建一个空仓库，例如 `prj003-face-counter`。不要勾选自动创建
README、`.gitignore` 或许可证，以免第一次推送产生冲突。

在项目目录打开 PowerShell：

```powershell
cd F:\ckp\graoh2
git init
git add .
git status
git commit -m "完成 PRJ003 课堂人脸计数系统"
git branch -M main
git remote add origin https://github.com/你的用户名/prj003-face-counter.git
git push -u origin main
```

执行 `git add .` 后务必先看 `git status`，确认没有 `.venv`、密钥或真实照片，再
提交。如果目录已经是 Git 仓库，跳过 `git init`；如果已经存在 `origin`，使用
`git remote set-url origin 新地址`，不要重复添加。

推送后在 GitHub 网页确认以下两点：

1. `models/face_detection_yunet_2023mar.onnx` 存在，大小约 227 KB，而不是几十字节
   的 Git LFS 指针。
2. `requirements.txt` 和 `.streamlit/config.toml` 位于仓库根目录。

## 部署到 Streamlit Community Cloud

1. 打开 <https://share.streamlit.io>，登录并连接 GitHub。
2. 点击右上角 **Create app**。
3. 选择 **Yup, I have an app**。
4. 选择 GitHub 仓库、`main` 分支，入口文件填写 `app.py`。
5. 在 **Advanced settings** 中选择 Python 3.12。本项目的锁定依赖支持该版本。
6. 本项目不使用密钥，Secrets 留空。
7. 可选填一个容易记忆的子域名，然后点击 **Deploy**。
8. 等待构建完成。成功后会获得形如 `https://你的名称.streamlit.app` 的公网地址。

如果构建失败，打开右侧部署日志，优先检查：

- 是否提交了 `requirements.txt`；
- ONNX 模型是否存在并完整；
- 是否意外使用了 `opencv-python` 而不是 `opencv-python-headless`；
- Community Cloud 的 Python 版本是否选择了 3.12；
- 仓库根目录与入口文件是否填写正确。

## 公网功能验收

建议使用不包含真实学生信息的演示图片完成以下检查：

1. 用电脑无痕窗口打开公网网址，确认无需本地文件路径即可加载。
2. 上传无脸图，确认预览正常，点击“开始检测”后显示 0，并能下载 PNG。
3. 上传单人图，确认显示 1 个框且数量为 1。
4. 上传多人图，核对可见矩形框数量与页面数字一致。
5. 下载标注图并打开，确认格式是 PNG、图片完整、边界框可见。
6. 用手机或浏览器窄窗口打开页面，确认标题、上传区、按钮和结果图不横向溢出。
7. 最后重新打开页面，确认上一位访问者的上传图片不会出现在新会话中。

## 本地离线备用方案

联网时先完成一次依赖和模型准备：

```powershell
cd F:\ckp\graoh2
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_model.py
```

模型和依赖准备完成后，断网也可启动：

```powershell
cd F:\ckp\graoh2
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器访问 `http://localhost:8501`。离线运行时程序直接读取项目内的 ONNX 模型，
不会尝试联网。

## PPT 现场演示清单

1. 提前打开公网地址，同时在本机准备好离线页面作为备用。
2. 准备三张脱敏演示图：无脸、单人、多人；不要使用未经授权的真实学生照片。
3. 先讲清楚模型来自 OpenCV Zoo、训练数据为 WIDER FACE，并非自行训练。
4. 上传多人图，展示原图预览。
5. 展开参数区域，简短说明置信度阈值，然后点击“开始检测”。
6. 指出大号人数、每个检测框以及原图/结果图的并排对比。
7. 点击下载按钮，打开结果 PNG。
8. 最后说明遮挡、低头和远距离小脸可能漏检，因此检测数不等于实际考勤人数。
