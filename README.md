# 智墨：书法家识别、训练与鉴赏 Agent

智墨是一个面向中文书法图像的识别与鉴赏原型项目。当前代码已经覆盖数据整理、书法家分类模型训练、单字/多字推理、Grad-CAM 可解释性、图像质量评估、书法知识库检索和 LangChain Agent 编排。

## 当前状态

- 项目主体位于 `zhimo/` 目录。
- 当前分类模型代码不再只面向早期 17 类数据集，`train.py` 和 `merge_split_data.py` 中的 `author_map` 已扩展到约 54 个书法家/作者 ID。
- 推理默认权重路径为 `zhimo/checkpoints/convnext.pth`。
- 训练默认数据目录为 `zhimo/dataset/`，结构为 `train/<作者ID>/图片` 和 `test/<作者ID>/图片`。
- 训练默认输出目录为 `zhimo/checkpoints_test_gelu/`。
- 当前源码中仍有部分中文注释/提示词存在编码乱码，但主要程序结构和运行路径可判断。

## 功能概览

- 数据整理：从多个原始数据集目录合并、去重、限量采样并划分训练/测试集。
- 模型训练：基于 `timm` backbone 训练书法家分类器，支持命令行覆盖训练参数。
- 单字识别：加载训练权重，输出书法家、置信度、全部类别概率和 backbone 信息。
- TTA 推理：支持原图、水平翻转、多尺度等测试时增强后平均概率。
- Grad-CAM 可解释性：可生成热力图并保存到 `image_test/cam_output.png`。
- 图像质量评估：评估泛黄、褪色、噪声、模糊等退化因素。
- 多字作品分析：传统图像处理切分多字作品，并逐字识别后投票。
- RAG 知识检索：使用 Chroma 和 Ollama `bge-m3` embedding 检索书法家知识。
- Agent 编排：通过 LangChain 工具组合识别、质量分析、知识检索和回答生成。

## 项目结构

```text
.
|-- README.md                    # 项目说明
`-- zhimo/
    |-- agent.py                 # LangChain Agent 示例入口
    |-- train.py                 # 书法家分类模型训练脚本
    |-- merge_split_data.py      # 多数据源合并、去重、限量采样和 train/test 划分
    |-- dataset_require.py       # HCSU 数据集下载示例，使用 Hugging Face datasets
    |-- calligrapher_recognizer.py # 模型加载、单张/批量推理、TTA、Grad-CAM
    |-- calligrapher_tool.py     # Agent 工具：识别、质量分析、RAG、多字分析
    |-- image_quality.py         # 图像质量评估
    |-- segment.py               # 多字作品传统切分方法
    |-- rag_setup.py             # Chroma 向量库初始化与加载
    |-- knowledge.py             # 书法家知识文本
    |-- adverse_aug.py           # 图像退化/增强相关脚本
    |-- convert_to_yolo.py       # YOLO 数据格式转换脚本
    |-- test_gam.py              # Grad-CAM 推理测试脚本
    |-- test_tool.py             # 工具调用测试脚本
    |-- yolo.py                  # 预留脚本
    |-- frontend/                # 本地网页前端和 aiohttp 服务
    |-- image_test/              # 示例图片、切分结果和 CAM 输出
    |-- chroma_db/               # 已持久化的 Chroma 数据库
    `-- requirements.txt         # Python 依赖
```

## 环境准备

建议在独立 conda 环境中安装依赖。进入 `zhimo/` 后执行：

```powershell
cd zhimo
conda activate practice
python -m pip install -r requirements.txt
```

如果当前 PowerShell 没有初始化 `conda`，可以直接调用 Anaconda 安装目录下的 `conda.exe`：

```powershell
cd zhimo
& 'D:\anaconda\Scripts\conda.exe' run -n practice python -m pip install -r requirements.txt
```

之前已在 `practice` 环境中验证过以下关键包可导入：

```text
torch 2.14.1+cpu
torchvision 0.29.1+cpu
langchain 1.4.3
chromadb 1.5.9
cv2 5.0.0
timm 1.0.30
```

注意：安装时 pip 曾提示 `torchaudio 2.5.1+cu121` 与 `torch 2.14.1` 存在版本冲突。当前项目代码没有使用 `torchaudio`，不影响现有识别流程；如果后续需要音频相关包，应单独处理该冲突。

## 数据准备

### 目录结构

训练脚本默认读取 `zhimo/dataset/`：

```text
zhimo/dataset/
|-- train/
|   |-- wxz/
|   |-- yzq/
|   `-- ...
`-- test/
    |-- wxz/
    |-- yzq/
    `-- ...
```

其中子目录名应使用 `train.py` / `merge_split_data.py` 中 `CONFIG["author_map"]` 定义的作者 ID，例如 `wxz`、`yzq`、`lgq` 等。

### 合并并划分数据

`merge_split_data.py` 会从多个源目录读取数据，按“书法家-字体/书体”文件夹归并，冲突时保留图片数量最多的目录，并对每个“书法家-字体/书体”最多保留 `200` 张图片。默认源目录包括：

```text
zhimo/dataset_total/dataset1
zhimo/dataset_total/dataset2
zhimo/dataset_total/dataset3
zhimo/dataset4
zhimo/dataset5
```

默认输出到：

```text
zhimo/dataset/train
zhimo/dataset/test
```

运行：

```powershell
cd zhimo
python merge_split_data.py
```

注意：如果目标目录已存在且非空，脚本会提示确认，并在确认后清空目标目录。运行前请确认没有需要保留的数据。

### HCSU 数据集示例

`dataset_require.py` 目前只是 Hugging Face 数据集下载示例：

```python
from datasets import load_dataset

ds = load_dataset("Tongji209/HCSU")
```

如需使用它，需要额外安装 `datasets` 包；当前 `requirements.txt` 中没有显式列出 `datasets`。

## 模型训练

默认训练配置在 `zhimo/train.py` 的 `CONFIG` 中，关键默认值包括：

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
cd zhimo
python train.py
```

也可以覆盖参数：

```powershell
cd zhimo
python train.py --epochs 10 --batch_size 32 --backbone convnext_tiny
```

断点恢复：

```powershell
cd zhimo
python train.py --resume ./checkpoints_test_gelu/calligrapher_classifier_e10.pth
```

训练输出包括：

```text
zhimo/checkpoints_test_gelu/calligrapher_classifier.pth      # 当前最佳模型
zhimo/checkpoints_test_gelu/calligrapher_classifier_e*.pth   # 每轮 checkpoint
zhimo/checkpoints_test_gelu/loss_accuracy_curve.png          # 损失/准确率曲线
zhimo/checkpoints_test_gelu/confusion_matrix.png             # 混淆矩阵
```

## 推理与鉴赏

### 本地网页前端

当前新增了一个独立的本地网页控制台，位置为：

```text
zhimo/frontend/
|-- server.py
`-- static/
    |-- index.html
    |-- styles.css
    `-- app.js
```

这个前端不包含训练功能，重点用于展示和调用推理期 Agent 工作流：

- 上传书法图片。
- 选择单字或多字作品模式。
- 展示本地 Agent 调用步骤：图片读取、图像质量评估、单字识别或多字切分、模型推理、知识库检索、综合回答。
- 展示质量指标、Top-K 候选、多字切分框、Grad-CAM 热力图和原始 JSON。
- 在缺少模型权重、缺少 Grad-CAM 依赖、Ollama/RAG 不可用时，返回可读诊断，不让页面崩溃。

启动方式：

```powershell
cd zhimo
python frontend/server.py
```

默认访问地址：

```text
http://127.0.0.1:8765
```

如果 `8765` 被占用，服务会自动选择一个可用端口，并在终端打印实际地址。

可选：通过环境变量指定权重路径：

```powershell
$env:ZHIMO_MODEL_PATH = "./checkpoints_test_gelu/calligrapher_classifier.pth"
python frontend/server.py
```

健康检查接口：

```text
http://127.0.0.1:8765/api/health
```

在当前缺少 `zhimo/checkpoints/convnext.pth` 的情况下，前端仍应能运行到模型推理入口。预期诊断为：

```text
已成功运行到模型推理入口，但未找到模型权重。补齐权重后可继续推理。
```

已做过的本地自检：

- `python -m py_compile frontend/server.py` 通过。
- `/api/health` 可返回模型权重缺失、Chroma 是否存在、样例图片列表等信息。
- 使用 `image_test/test.png` 调用单字模式：图片读取和质量评估完成，随后在模型加载阶段返回缺权重诊断。
- 使用 `image_test/test_multi.png` 调用多字模式：图片读取、质量评估和字符切分完成，随后在逐字识别阶段返回缺权重诊断。

### 模型权重

`CalligrapherRecognizer` 当前默认权重路径为：

```text
zhimo/checkpoints/convnext.pth
```

如果使用 `train.py` 的默认输出，需要二选一：

- 将最佳权重复制/重命名到 `zhimo/checkpoints/convnext.pth`。
- 或在代码中初始化时显式传入 `model_path="./checkpoints_test_gelu/calligrapher_classifier.pth"`。

示例：

```python
from calligrapher_recognizer import CalligrapherRecognizer

recognizer = CalligrapherRecognizer(
    model_path="./checkpoints_test_gelu/calligrapher_classifier.pth"
)
```

### 单张图片识别

`calligrapher_recognizer.py` 支持：

```python
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

### Grad-CAM 测试

`test_gam.py` 默认读取：

```text
zhimo/image_test/test.png
```

并将 CAM 结果保存到：

```text
zhimo/image_test/cam_output.png
```

运行：

```powershell
cd zhimo
python test_gam.py
```

注意：`predict_with_cam` 依赖 `pytorch-grad-cam`。当前 `requirements.txt` 中没有显式列出该包；如果运行时报 `No module named pytorch_grad_cam`，需要额外安装：

```powershell
python -m pip install grad-cam
```

### Agent 示例

`agent.py` 默认读取 `./image_test/test1.png`，将图片转为 base64 后交给 Agent：

```powershell
cd zhimo
python agent.py
```

Agent 可调用以下工具：

- `analyze_calligraphy`：单字识别，并附带图像质量报告。
- `analyze_multi_char`：多字作品切分、逐字识别和投票。
- `identify_calligrapher`：仅执行书法家识别。
- `search_knowledge`：检索书法家风格知识。

### 多字切分测试

`segment.py` 默认读取 `./image_test/image.png`，并保存三种切分方式的可视化结果：

```powershell
cd zhimo
python segment.py
```

预期输出包括：

```text
result_cc.png
result_proj.png
result_auto.png
```

## RAG 知识库

仓库中已有持久化的 `zhimo/chroma_db/`。如果需要初始化、检查或重建向量库，可运行：

```powershell
cd zhimo
python rag_setup.py
```

`rag_setup.py` 使用 `langchain_ollama.OllamaEmbeddings(model="bge-m3")`。运行前需要本机安装并启动 Ollama，且已拉取模型：

```powershell
ollama pull bge-m3
```

## 大模型 API Key

`agent.py` 通过 LangChain 初始化 `deepseek:deepseek-flash`。建议使用 `.env` 或系统环境变量配置对应 API Key，不要把密钥提交到仓库。

示例 `.env`：

```env
DEEPSEEK_API_KEY=你的密钥
```

## 注意事项

- 训练、数据整理和知识库脚本都假设当前工作目录是 `zhimo/`，建议先 `cd zhimo` 再运行。
- 当前部分源码注释和提示词存在中文乱码；如果要展示给用户，建议统一修复为 UTF-8 文本。
- `requirements.txt` 对依赖版本固定较严格；如果 pip 提示某些版本不可用，需要根据 Python 版本和包索引调整。
- `dataset_require.py` 需要 `datasets` 包，`test_gam.py` 需要 `grad-cam` 包，这两个依赖当前没有显式写入 `requirements.txt`。
- `test_tool.py` 默认读取 `test.jpg`，当前工作区未包含该文件；运行前需要改为实际图片路径。
- 识别结论依赖模型权重、图片质量和切分效果。多字作品中如果切分数量过少或投票结果分散，应谨慎解释最终判断。

## 典型工作流

1. 进入 `zhimo/` 并安装依赖。
2. 准备原始数据目录，运行 `python merge_split_data.py` 生成 `dataset/train` 和 `dataset/test`。
3. 运行 `python train.py` 训练书法家分类模型。
4. 将最佳权重用于推理，或放到 `checkpoints/convnext.pth`。
5. 启动 Ollama，并确认 `bge-m3` 可用。
6. 在 `.env` 或系统环境变量中配置大模型 API Key。
7. 运行 `python rag_setup.py` 检查向量知识库。
8. 运行 `python agent.py` 或 `python test_gam.py` 做识别、鉴赏和可解释性测试。
