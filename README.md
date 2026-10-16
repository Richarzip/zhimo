# 智墨：书法家识别与鉴赏 Agent

智墨是一个面向中文书法图像的本地应用，提供书法家识别（ConvNeXt + Swin 双模型软投票集成）、多字作品分析、图像质量评估、Grad-CAM 可解释分析、书法知识检索（RAG）与多轮 Agent 对话，并支持将结果导出为 PDF 报告。项目包含 `aiohttp` 网页服务，以及可单独调用的 Python 包接口。

## 快速开始

以下命令均在项目根目录 `zhimo/` 下执行。

### 安装依赖

项目使用 `src` 布局，需要 Python 3.10+：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
# 训练脚本额外依赖
python -m pip install -e ".[train]"
```

### 配置权重

默认集成推理需要两个权重文件：

```text
checkpoints/
|-- convnext.pth
`-- swin.pth
```

```powershell
# Windows 用分号分隔，Linux/macOS 用冒号
$env:ZHIMO_MODEL_PATHS = ".\checkpoints\convnext.pth;.\checkpoints\swin.pth"
```

单模型诊断可设置 `ZHIMO_MODEL_PATH`。权重、`.env`、Chroma 数据库和数据集不提交到仓库。

### 启动网页

```powershell
python apps/web/server.py
```

默认地址为 `http://127.0.0.1:8765`。主要接口：

```text
GET  /              网页
GET  /api/health    运行状态、权重和知识库检查
POST /api/analyze   单字、多字或对话分析
POST /api/export    根据最近一次图片鉴别结果生成 PDF 报告
```

上传限制：单张图片最大 32 MB、最大 25,000,000 像素，支持 JPEG、PNG、WEBP、BMP 和 GIF。

### Agent 配置

Agent 使用的 API 配置放在 `.env` 中：

```env
DEEPSEEK_API_KEY=your_key_here
```

不要把真实密钥提交到 Git。

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
|-- training/              数据整理和训练
|-- evaluation/            集成评估
|-- research/              资料审计和来源记录
|-- tests/unit/            Python 单元测试
|-- frontend/tests/        前端与 HTTP 回归测试
|-- agent.py               一次性 Agent 命令行示例
|-- pyproject.toml         Python 包配置
`-- requirements.txt       完整运行依赖
```

## Python 接口

安装后可直接导入；未安装时可临时设置 `PYTHONPATH`：

```python
from PIL import Image
from zhimo.application import get_recognizer, resolve_model_paths

recognizer = get_recognizer()
result = recognizer.recognize(Image.open("image_test/test.png"))
print(result["calligrapher"], result["confidence"])
```

Agent 使用统一构建入口：

```python
from zhimo.agent import create_calligraphy_agent
agent = create_calligraphy_agent()
```

## 训练与评估

```powershell
python training/merge_split_data.py        # 数据切分
python training/train.py --backbone convnext_tiny
python evaluation/evaluate_ensemble.py --data-root .\dataset\test --n 120 --seed 42
```

训练不是 Web 运行的必要步骤；已有推理权重时可以跳过。

## RAG 知识库

知识库默认保存在 `chroma_db/`，embedding 使用 Ollama 的 `bge-m3`：

```powershell
ollama pull bge-m3
ollama serve
python -m zhimo.knowledge.chroma
```

网页中的"知识检索"开关默认关闭，RAG 不可用时普通识别仍可运行。

## 测试

```powershell
python -B -m unittest discover -s tests -p "test_*.py" -v
python -B -m unittest discover -s frontend/tests -p "test_*.py" -v
node --test frontend/tests/test_app.cjs
```

测试会替换模型和知识库依赖，不需要权重、Ollama 或真实 API key。

## 主要配置项

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `ZHIMO_MODEL_PATHS` | 多模型权重路径 | `checkpoints/convnext.pth` 和 `checkpoints/swin.pth` |
| `ZHIMO_MODEL_PATH` | 单模型诊断路径 | 未设置 |
| `ZHIMO_CHROMA_DIR` | Chroma 数据目录 | `chroma_db/` |
| `ZHIMO_HOST` / `ZHIMO_PORT` | Web 监听地址/端口 | `127.0.0.1` / `8765` |
| `DEEPSEEK_API_KEY` | Agent API key | 未设置 |

## 常见问题

- **`Import "zhimo..." could not be resolved`**：在 VS Code 中打开 `zhimo/` 作为工作区根目录并选择安装依赖的解释器；项目导入名是 `zhimo`，不是 `src.zhimo`。
- **缺少模型权重**：检查 `/api/health` 返回的 `model_paths` 是否指向实际存在的 `.pth` 文件；双模型模式要求两个权重类别顺序一致。
- **RAG 或 Agent 不可用**：确认 Ollama 已启动并拥有 `bge-m3`，再检查 `.env` 中的 API key；RAG/Agent 故障不影响普通识别。
