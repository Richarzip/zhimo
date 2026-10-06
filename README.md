# 智墨：书法家识别与鉴赏 Agent

智墨是一个面向中文书法图像的本地应用，提供书法家识别、多字作品分析、图像质量评估、Grad-CAM、书法知识检索和 Agent 对话。项目包含 `aiohttp` 网页服务，以及可单独调用的 Python 包接口。

## 当前状态

- 正式 Python 包位于 `src/zhimo/`。
- Web 推荐入口是 `apps/web/server.py`。
- 默认加载 `checkpoints/convnext.pth` 和 `checkpoints/swin.pth`，对两个模型的 softmax 概率取平均。
- 权重、`.env`、Chroma 数据库和数据集不提交到仓库。
- 训练代码位于 `training/`，评估代码位于 `evaluation/`，资料审计代码位于 `research/`。
- 根目录只保留 `agent.py` 一次性 Agent 示例，不再保留旧的兼容脚本。

## 快速开始

以下命令均在项目根目录 `zhimo/` 下执行。

### 安装依赖

项目使用 `src` 布局。推荐使用已有的 Conda 环境 `practice`，或创建 Python 3.10+ 环境：

```powershell
conda activate practice
python -m pip install -r requirements.txt
python -m pip install -e .
```

训练脚本额外需要的绘图库和指标库：

```powershell
python -m pip install -e ".[train]"
```

### 配置权重

默认集成推理需要：

```text
checkpoints/
|-- convnext.pth
`-- swin.pth
```

Windows PowerShell 使用分号分隔多个权重路径；Linux/macOS 使用冒号：

```powershell
$env:ZHIMO_MODEL_PATHS = ".\checkpoints\convnext.pth;.\checkpoints\swin.pth"
```

临时单模型诊断：

```powershell
$env:ZHIMO_MODEL_PATH = ".\checkpoints\convnext.pth"
```

### 启动网页

```powershell
python apps/web/server.py
```

默认地址为 `http://127.0.0.1:8765`。指定地址和端口：

```powershell
python apps/web/server.py --host 127.0.0.1 --port 8766
```

接口：

```text
GET  /              网页
GET  /api/health    运行状态、权重和知识库检查
POST /api/analyze   单字、多字或对话分析
```

上传限制：单张图片最大 `32 MB`，最大 `25,000,000` 像素，支持 JPEG、PNG、WEBP、BMP 和 GIF。

## 项目结构

```text
zhimo/
|-- src/zhimo/             正式 Python 包
|   |-- application/       应用服务、分析流程、输出和 PDF 报告
|   |-- agent/             Agent 构建和工具
|   |-- config/            模型、Web 和 Chroma 配置
|   |-- knowledge/         知识数据、仓储和 Chroma 适配器
|   `-- vision/            识别器、集成推理、预处理、质量和切分
|-- apps/web/server.py     推荐 Web 启动入口
|-- frontend/              网页静态资源和 HTTP 服务实现
|-- training/              数据整理、训练和格式转换
|-- evaluation/            集成评估和 Grad-CAM/工具脚本
|-- research/              资料审计和来源记录
|-- tests/unit/            应用、Agent、知识审计和视觉单元测试
|-- frontend/tests/        前端与 HTTP 回归测试
|-- checkpoints/           本地模型权重，不提交
|-- chroma_db/             本地 Chroma 数据库
|-- image_test/            示例图片和调试输出
|-- agent.py               一次性 Agent 命令行示例
|-- pyproject.toml         Python 包配置
`-- requirements.txt       完整运行依赖
```

## Python 接口

安装项目后可以直接导入；未安装时可临时设置：

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
```

应用层会根据配置自动选择单模型或集成模型：

```python
from PIL import Image
from zhimo.application import get_recognizer, resolve_model_paths

print(resolve_model_paths())
recognizer = get_recognizer()
result = recognizer.recognize(Image.open("image_test/test.png"))
print(result["calligrapher"], result["confidence"])
```

直接使用集成识别器：

```python
from PIL import Image
from zhimo.vision import EnsembleRecognizer

result = EnsembleRecognizer().recognize(
    Image.open("image_test/test.png"), tta=True
)
```

Agent 使用统一构建入口：

```python
from zhimo.agent import create_calligraphy_agent

agent = create_calligraphy_agent()
```

## Agent 示例

`agent.py` 只有在直接执行时才初始化模型和 Agent：

```powershell
python agent.py
python agent.py image_test/test.png --prompt "请分析这幅字的风格和可能的书法家"
```

主要工具包括 `identify_calligrapher`、`analyze_calligraphy`、`analyze_multi_char` 和 `search_knowledge`。

Agent 使用的 API 配置放在 `.env` 中，例如：

```env
DEEPSEEK_API_KEY=your_key_here
```

不要把真实密钥提交到 Git。

## RAG 知识库

知识库默认保存在 `chroma_db/`，embedding 使用 Ollama 的 `bge-m3`：

```powershell
ollama pull bge-m3
ollama serve
python -m zhimo.knowledge.chroma
```

RAG 不可用时，普通图像识别仍可运行；Web 和 Agent 会返回结构化诊断。

## 训练和评估

训练数据默认结构：

```text
dataset/
|-- train/<author>/...
`-- test/<author>/...
```

准备数据、训练和断点恢复：

```powershell
python training/merge_split_data.py
python training/train.py --help
python training/train.py
python training/train.py --epochs 10 --batch_size 32 --backbone convnext_tiny
python training/train.py --resume .\checkpoints_test_gelu\calligrapher_classifier_e10.pth
```

训练输出默认写入 `checkpoints_test_gelu/`。训练不是 Web 运行和普通识别的必要步骤；已有推理权重时可以跳过。

集成评估：

```powershell
python evaluation/evaluate_ensemble.py --data-root .\dataset\test --n 120 --seed 42
```

结果默认写入项目根目录 `evaluation_result.txt`，也可以通过 `--out` 指定路径。

## 研究资料与来源审计

研究脚本和审计记录位于 `research/`：

```powershell
python research/baidu_knowledge_audit.py
python research/research_assets.py
```

当前应用优先使用 `research/baidu_knowledge_audit.json`、`research/knowledge_audit.json` 和 `frontend/static/assets/` 中经过审计的素材。证据不足时会保留为待人工复核，不会直接作为已验证作品展示。

## 测试

全部 Python 单元测试：

```powershell
python -B -m unittest discover -s tests -p "test_*.py" -v
```

前端和 HTTP 回归测试：

```powershell
python -B -m unittest discover -s frontend/tests -p "test_*.py" -v
node --test frontend/tests/test_app.cjs
```

测试会替换模型和知识库依赖，不需要模型权重、运行中的 Ollama 或真实 API key。真实全流程测试才需要这些外部服务。

## 主要配置项

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `ZHIMO_MODEL_PATHS` | 多模型权重路径 | `checkpoints/convnext.pth` 和 `checkpoints/swin.pth` |
| `ZHIMO_MODEL_PATH` | 单模型诊断路径 | 未设置 |
| `ZHIMO_CHROMA_DIR` | Chroma 数据目录 | `chroma_db/` |
| `ZHIMO_HOST` | Web 默认监听地址 | `127.0.0.1` |
| `ZHIMO_PORT` | Web 默认端口 | `8765` |
| `DEEPSEEK_API_KEY` | Agent API key | 未设置 |

## 常见问题

### `Import "zhimo..." could not be resolved`

请在 VS Code 中打开 `zhimo/` 作为工作区根目录，并选择安装项目依赖的 Python 解释器。项目使用 `src` 布局，正式包路径是 `src/zhimo/`，导入名仍然是 `zhimo`，不是 `src.zhimo`。也可以临时设置：

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
```

### 缺少模型权重

检查 `/api/health` 返回的 `model_paths` 是否指向实际存在的 `.pth` 文件。双模型模式要求两个权重类别顺序一致。

### RAG 或 Agent 不可用

确认 Ollama 已启动并拥有 `bge-m3`，再检查 `.env` 中的 API key。RAG/Agent 故障不应影响不使用这些功能的普通识别。

### `predict_with_cam` 缺少依赖

```powershell
python -m pip install grad-cam
```

普通识别不依赖 Grad-CAM。