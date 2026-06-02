# LocateAnything-3B

LocateAnything-3B 是一个基于 Transformer 架构的视觉定位模型，能够根据文本描述在图像中定位目标物体。

## 模型下载

使用 ModelScope 下载模型：

```bash
# 安装 modelscope
pip install modelscope

# 下载模型（约3GB）
modelscope download --model nv-community/LocateAnything-3B
```

模型默认下载路径：`~/.cache/modelscope/hub/models/nv-community/LocateAnything-3B`

## 环境依赖

### 关键版本要求

| 库名称 | 版本要求 | 说明 |
|--------|----------|------|
| Python | >= 3.10 | 建议使用 Anaconda 虚拟环境 |
| transformers | >=4.40.0, <4.48.0 | **重要**：4.48+ 版本移除了 `all_tied_weights_keys` |
| torch | >= 2.1.0 | GPU 环境建议使用 CUDA 12.x |
| torchvision | >= 0.16.0 | 与 PyTorch 版本匹配 |
| numpy | >= 1.25.0, <2.0.0 | 避免 numpy 2.0 兼容性问题 |
| Pillow | >= 11.1.0 | 图像处理核心库 |
| opencv-python-headless | == 4.11.0.86 | 固定版本确保稳定性 |

### 安装依赖

```bash
# 克隆仓库
git clone <repository-url>
cd LocateAnything-3B

# 创建虚拟环境（推荐）
conda create -n locate3b python=3.10
conda activate locate3b

# 安装依赖
pip install -r requirements.txt
```

### 可选依赖

```bash
# 用于 PEFT 微调
pip install peft>=0.10.0

# 用于视频处理
pip install decord>=0.6.0

# 用于大规模数据处理
pip install lmdb>=1.7.5

# 用于分布式训练/推理
pip install accelerate>=0.25.0
```

## 快速开始

### 1. 模型测试

```bash
# 运行基础测试（验证模型加载）
python simple_test.py
```

### 2. 使用 GUI 界面

```bash
# 安装 PyQt5（如果未安装）
pip install pyqt5

# 启动图形界面
python gui_simple.py
```

### 3. Python API 示例

```python
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel

# 模型路径
MODEL_PATH = "~/.cache/modelscope/hub/models/nv-community/LocateAnything-3B"

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
    device_map=device,
).eval()

# 推理
# 请参考官方文档或源码中的推理示例
```

## 目录结构

```
LocateAnything-3B/
├── gui_simple.py          # 图形界面工具
├── simple_test.py         # 模型加载测试脚本
├── test_cars.py           # 汽车定位测试示例
└── requirements.txt       # 依赖列表
```

## 功能特性

- ✅ 文本引导的图像目标定位
- ✅ 支持中文和英文目标描述
- ✅ 可视化结果展示
- ✅ 结果图片保存（自动命名：[原文件名]_[时间戳].png）
- ✅ 图片缩放查看（25% - 300%）

## 常见问题

### 1. 版本兼容性问题

**问题**：`AttributeError: 'Qwen2ForCausalLM' object has no attribute 'all_tied_weights_keys'`

**解决方案**：确保 `transformers` 版本低于 4.48.0

```bash
pip install "transformers>=4.40.0,<4.48.0"
```

### 2. CUDA 内存不足

**解决方案**：使用 CPU 推理或调整 batch size

```python
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,  # 使用 float32 减少显存占用
    trust_remote_code=True,
    device_map="cpu",
).eval()
```

### 3. 中文显示问题

GUI 界面已内置中文支持，会自动检测系统中的中文字体：
- Noto Sans CJK SC (思源黑体)
- WenQuanYi MicroHei (文泉驿微米黑)
- SimHei (黑体)

如果系统没有中文字体，可安装：

```bash
sudo apt install fonts-noto-cjk  # Ubuntu/Debian
```

## 性能要求

| 配置 | 最小要求 | 推荐配置 |
|------|----------|----------|
| GPU 显存 | 8GB | 16GB+ |
| 内存 | 16GB | 32GB+ |
| CUDA | 11.8+ | 12.x |

## 许可证

请参考 ModelScope 上的模型许可证。

## 引用

如果使用此模型，请引用原作者的工作。