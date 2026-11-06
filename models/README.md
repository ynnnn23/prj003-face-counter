# 模型目录

本项目使用 OpenCV Zoo 提供的 YuNet 预训练人脸检测模型：

- 文件名：`face_detection_yunet_2023mar.onnx`
- 来源：<https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- 原始训练项目：<https://github.com/ShiqiYu/libfacedetection.train>
- 许可证：模型随 OpenCV Zoo 按 Apache License 2.0 发布

该模型是第三方预训练模型，并非本课程项目自行训练。若模型文件缺失，请在项目
根目录执行：

```powershell
python scripts/download_model.py
```
