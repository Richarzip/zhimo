# 智墨：书法家识别、训练与鉴赏 Agent

智墨是一个面向中文书法图像的本地原型项目，覆盖书法家分类模型训练、单字/多字推理、Grad-CAM 可解释性、图像质量评估、书法知识库检索，以及基于 LangChain/LangGraph 的 Agent 编排。项目还包含一个 `aiohttp` 本地网页前端，用于上传图片并查看识别、切分、质量评估、知识检索和诊断结果。

## 当前状态

- 项目根目录就是当前 `zhimo/` 目录。
- 训练脚本默认读取 `./dataset/`，输出到 `./checkpoints_test_gelu/`。
- 前端和 Agent 默认使用 ConvNeXt + Swin 双模型软投票集成，权重路径为 `./checkpoints/convnext.pth` 和 `./checkpoints/swin.pth`。
- 如果使用训练脚本默认产物，需要把对应权重复制/指定给推理端；临时诊断时也可用单模型覆盖。
- 当前代码面向约 54 个书法家/作者 ID，映射定义在 `train.py` 和 `merge_split_data.py` 的 `CONFIG["author_map"]` 中。
- 部分源码注释和服务端提示文本存在编码乱码，但主要运行入口、接口和数据流仍可判断。

## 功能概览

- 数据整理：合并多个原始数据源，按作者归并、去重、限量采样并划分训练/测试集。
- 模型训练：基于 `timm` backbone 训练书法家分类器，支持命令行覆盖训练参数和断点恢复。
- 单字识别：加载 `.pth` 权重，输出书法家、置信度、全部类别概率、backbone 等信息。
- TTA 推理：支持原图、水平翻转、多尺度等测试时增强后平均概率。
- Grad-CAM：可生成关注热力图，缺少依赖时不影响普通识别。
- 多字作品分析：使用传统图像处理切分多字作品，再逐字识别并投票。
- 图像质量评估：评估泛黄、褪色、噪声、模糊等退化因素。
- RAG 知识检索：使用 Chroma 持久化向量库和 Ollama `bge-m3` embedding 检索书法知识。
- Agent 编排：组合识别、质量分析、多字分析、知识检索和回答生成。
- 本地网页：上传图片，选择单字/多字/对话模式，查看步骤、Top-K、切分框、CAM、JSON 和错误诊断。

## 项目结构

```text
zhimo/
|-- README.md                       # 项目说明
|-- requirements.txt                # Python 依赖
|-- train.py                        # 书法家分类模型训练脚本
|-- merge_split_data.py             # 数据合并、去重、采样和 train/test 划分
|-- calligrapher_recognizer.py      # 模型加载、单张/批量推理、TTA、Grad-CAM
|-- ensemble_recognizer.py          # 集成识别器
|-- calligrapher_tool.py            # Agent 工具：识别、质量分析、RAG、多字分析
|-- agent.py                        # Agent 示例入口
|-- agent_builder.py                # LangGraph Agent 构建
|-- image_preprocess.py             # 图像预处理，例如黑底白字反色修正
|-- image_quality.py                # 图像质量评估
|-- segment.py                      # 多字作品传统切分方法
|-- rag_setup.py                    # Chroma 向量库初始化与加载
|-- knowledge.py                    # 书法家知识文本
|-- adverse_aug.py                  # 图像退化/增强脚本
|-- convert_to_yolo.py              # YOLO 数据格式转换脚本
|-- evaluate_ensemble.py            # 集成模型评估
|-- evaluate_result.txt             # 评估结果记录
|-- test_cam.py                     # Grad-CAM 推理测试脚本
|-- test_tool.py                    # 工具调用测试脚本
|-- frontend/
|   |-- server.py                   # aiohttp 本地服务
|   |-- static/                     # 前端页面、样式和脚本
|   `-- tests/                      # 前端与服务端回归测试
|-- image_test/                     # 示例图片、切分结果和 CAM 输出
`-- chroma_db/                      # 已持久化的 Chroma 数据库
```

## 环境准备

建议在独立环境中安装依赖。以下命令都假设当前目录是 `zhimo/`：

```powershell
python -m pip install -r requirements.txt
```

如果使用已有 conda 环境，例如 `practice`：

```powershell
conda activate practice
python -m pip install -r requirements.txt
```

当前依赖文件包含 `torch`、`torchvision`、`timm`、`aiohttp`、`Pillow`、`opencv-python`、`chromadb`、`langchain`、`langgraph`、`langchain-ollama`、`langchain-deepseek` 等核心包。

可选依赖说明：

- `predict_with_cam` 需要 `grad-cam` 包。如果缺失，普通识别仍可运行。
- `dataset_require.py` 使用 Hugging Face `datasets`，但 `requirements.txt` 当前没有显式列出该包。
- RAG 检索需要本机 Ollama 可用，并已拉取 `bge-m3`。

## 本地网页前端

启动服务：

```powershell
python frontend/server.py
```

默认地址：

```text
http://127.0.0.1:8765
```

如果 `8765` 被占用，服务会自动选择一个可用端口，并在终端打印实际地址。也可以手动指定：

```powershell
python frontend/server.py --host 127.0.0.1 --port 8766
```

前端服务接口：

```text
GET  /              # 页面
GET  /api/health    # 健康检查
POST /api/analyze   # 图片分析
```

可通过环境变量指定集成模型权重，多个路径用系统路径分隔符连接。Windows PowerShell 示例：

```powershell
$env:ZHIMO_MODEL_PATHS = "./checkpoints/convnext.pth;./checkpoints/swin.pth"
python frontend/server.py
```

兼容旧的单模型诊断方式：设置 `ZHIMO_MODEL_PATH` 后，前端会退回单个 `CalligrapherRecognizer`。

上传限制和行为：

- 单张图片最大约 `32 MB`。
- 最大像素数为 `25,000,000`。
- 支持 `JPEG`、`PNG`、`WEBP`、`BMP`、`GIF`。
- 服务端一次只处理一个分析任务；并发上传会返回 busy 诊断。
- 模型权重缺失、Grad-CAM 依赖缺失、Ollama/RAG 不可用时，会返回结构化诊断，页面不应崩溃。

## 模型权重

前端和 Agent 默认使用 `EnsembleRecognizer`，读取两个权重并做 soft voting：

```text
./checkpoints/convnext.pth
./checkpoints/swin.pth
```

两个模型分别输出 softmax 概率后，对同一类别顺序的概率向量取平均，再选择平均概率最高的书法家。返回结果会包含 `ensemble.strategy`、`ensemble.members` 和 `ensemble.agreement`。

单次训练脚本默认最佳模型输出为：

```text
./checkpoints_test_gelu/calligrapher_classifier.pth
```

因此推理前需要准备对应权重：

- 将 ConvNeXt 权重放到 `./checkpoints/convnext.pth`。
- 将 Swin 权重放到 `./checkpoints/swin.pth`。
- 或通过 `ZHIMO_MODEL_PATHS` 显式指定两个权重路径。

代码示例：

```python
from ensemble_recognizer import EnsembleRecognizer

recognizer = EnsembleRecognizer(
    model_paths=["./checkpoints/convnext.pth", "./checkpoints/swin.pth"]
)
```

## 数据准备

训练脚本默认数据结构：

```text
dataset/
|-- train/
|   |-- wxz/
|   |-- yzq/
|   `-- ...
`-- test/
    |-- wxz/
    |-- yzq/
    `-- ...
```

子目录名应使用 `CONFIG["author_map"]` 中定义的作者 ID，例如 `wxz`、`yzq`、`lgq` 等。

合并并划分数据：

```powershell
python merge_split_data.py
```

`merge_split_data.py` 默认从多个源目录读取数据，并输出到：

```text
dataset/train
dataset/test
```

注意：如果目标目录已存在且非空，脚本会提示确认，并在确认后清空目标目录。运行前请确认没有需要保留的数据。

## 模型训练

默认训练配置在 `train.py` 的 `CONFIG` 中，关键默认值包括：

```text
data_root: ./dataset
output_dir: ./checkpoints_test_gelu
backbone: convnext_tiny
epochs: 50
batch_size: 64
lr: 8e-5
weight_decay: 5e-2
early_stop_patience: 10
```

运行训练：

```powershell
python train.py
```

覆盖参数：

```powershell
python train.py --epochs 10 --batch_size 32 --backbone convnext_tiny
```

断点恢复：

```powershell
python train.py --resume ./checkpoints_test_gelu/calligrapher_classifier_e10.pth
```

训练输出包括：

```text
checkpoints_test_gelu/calligrapher_classifier.pth
checkpoints_test_gelu/calligrapher_classifier_e*.pth
checkpoints_test_gelu/loss_accuracy_curve.png
checkpoints_test_gelu/confusion_matrix.png
```

## 推理与鉴赏

### 单张图片识别

```python
from PIL import Image
from ensemble_recognizer import EnsembleRecognizer

recognizer = EnsembleRecognizer()
image = Image.open("./image_test/test.png")

result = recognizer.recognize(image, tta=False)
result_tta = recognizer.recognize(image, tta=True)
```

返回字段包括：

```text
calligrapher
confidence
all_probabilities
num_classes
model_backbone
```

### Grad-CAM

`test_cam.py` 默认用于 Grad-CAM 推理测试。也可以直接调用：

```python
result = recognizer.predict_with_cam(image, tta=True)
```

如果运行时报 `No module named 'pytorch_grad_cam'`，安装：

```powershell
python -m pip install grad-cam
```

### Agent 示例

```powershell
python agent.py
```

Agent 工具主要包括：

- `analyze_calligraphy`：单字识别，并附带图像质量报告。
- `analyze_multi_char`：多字作品切分、逐字识别和投票。
- `identify_calligrapher`：仅执行书法家识别。
- `search_knowledge`：检索书法家风格知识。

### 多字切分

`segment.py` 默认读取 `./image_test/image.png`，并保存三种切分方式的可视化结果：

```powershell
python segment.py
```

预期输出包括：

```text
image_test/result_cc.png
image_test/result_proj.png
image_test/result_auto.png
```

## RAG 知识库

仓库中已有持久化的 `chroma_db/`。如果需要初始化、检查或重建向量库：

```powershell
python rag_setup.py
```

`rag_setup.py` 使用 `langchain_ollama.OllamaEmbeddings(model="bge-m3")`。运行前需要本机安装并启动 Ollama，且已拉取模型：

```powershell
ollama pull bge-m3
```

## 大模型配置

Agent 使用 DeepSeek/LangChain 相关组件。建议通过 `.env` 或系统环境变量配置 API Key，不要把密钥提交到仓库。

示例 `.env`：

```env
DEEPSEEK_API_KEY=你的密钥
```

## 测试

前端测试位于 `frontend/tests/`。在项目根目录运行：

```powershell
node --test frontend/tests/test_app.cjs
python -B -m unittest discover -s frontend/tests -p "test_*.py" -v
```

测试覆盖重点：

- `test_app.cjs`：前端脚本、DOM 交互、上传限制、错误展示和响应处理。
- `test_recognizer.py`：TTA 概率平均、Grad-CAM 类别对齐、CAM 失败回退等推理逻辑。
- `test_server.py`：本地 HTTP 接口、上传校验、并发限制、RAG/TTA 参数传递等服务端行为。

这些测试主要验证接口契约和回归行为，不验证真实模型准确率。

## 典型工作流

1. 安装依赖。
2. 准备原始数据目录，运行 `python merge_split_data.py` 生成 `dataset/train` 和 `dataset/test`。
3. 运行 `python train.py` 训练书法家分类模型。
4. 将最佳权重用于推理，或放到 `checkpoints/convnext.pth`。
5. 启动 Ollama，并确认 `bge-m3` 可用。
6. 配置大模型 API Key。
7. 运行 `python rag_setup.py` 检查向量知识库。
8. 运行 `python frontend/server.py` 使用网页，或运行 `python agent.py`、`python test_cam.py` 做脚本测试。

## 注意事项

- 建议在 `zhimo/` 目录下运行脚本；很多相对路径都按项目根目录设计。
- `.gitignore` 已忽略 `.env`、`.venv`、`checkpoints/`、`*.pth`、`API.txt`、`__pycache__/`、`*.pyc`。
- 当前仓库未包含默认推理权重 `checkpoints/convnext.pth`。
- `test_tool.py` 默认读取的图片路径可能需要按本机实际文件调整。
- 识别结论依赖模型权重、图片质量和切分效果。多字作品中如果切分数量过少或投票结果分散，应谨慎解释最终判断。
