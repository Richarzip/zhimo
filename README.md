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
POST /api/export    根据最近一次图片鉴别结果生成 PDF 报告
```

网页的运行环境面板只展示权重、Chroma 和 Python 状态，不再展示本地样例图文件名；`/api/health` 中保留的样例图字段主要供回归测试和调试使用。

上传限制：单张图片最大 `32 MB`，最大 `25,000,000` 像素，支持 JPEG、PNG、WEBP、BMP 和 GIF。`/api/export` 需要上传图片和已有分析结果，服务端会校验图片大小和格式；该接口不会重新运行识别。

### PDF 报告

PDF 由 `src/zhimo/application/report.py` 使用 ReportLab 排版和分页，Pillow 负责图片解码与格式转换。生成单次鉴别报告需要 ReportLab（`requirements.txt` 与包运行依赖均已声明），不需要额外安装字体：程序优先使用常见系统中文字体，找不到时回退到 ReportLab 的 `STSong-Light` CID 字体映射。报告包含鉴别摘要和模型结果；有图像时还会呈现上传图、模型输入图、质量指标，以及可用的 Grad-CAM、知识依据和已审核作品。报告中的图片仅为版面缩略图，不会改变识别输入。

前端“输出鉴别报告”选项会让 `/api/analyze` 在当前分析响应中附带 PDF 并自动下载。对话区“导出为PDF”则选择最近一次已完成的带图分析，将图片和已有结果提交给 `/api/export` 重新排版；它不是整段聊天记录导出，也不会重新识别，因此该路径中的原图和处理图使用同一张图片。普通分析不开启 PDF 时不会生成报告。

报告会清理常见 Markdown 标记，将标题、列表、引用、链接、强调和表格转为可读文本。PDF 文本在 ReportLab 中按段落流式布局，可自动换页；超长质量指标会拆分成可分页的表格行。中文字体是否可嵌入以及字形外观仍取决于运行环境。模型输出只是辅助分析，不构成艺术史鉴定结论。

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

## 图片自动去噪

网页默认开启“自动去噪”。单字、多字和 Agent 上传流程共用
`zhimo.vision.denoise.denoise_image`，使用本地 OpenCV，不需要额外权重或 API。
关闭开关会跳过去噪；HTTP 的 `denoise` 字段接受 `true` / `false`，省略时默认开启。
上传时先按 EXIF 方向摆正图片，关闭去噪也会处理方向。

默认 `adaptive` 模式先诊断，再按需组合以下步骤：

| 检测情况 | 处理方式与限制 |
| --- | --- |
| 无明显可处理退化，或短边不足 16 像素 | 返回摆正方向后的 RGB 原图，不做滤波 |
| 孤立脉冲噪点 | 仅替换明显偏离周围像素的点，取 3×3 中值；至少 8 点且占比达到 0.05% 才启用 |
| 平坦区域存在连续噪声 | 双边滤波，强度随估计噪声调整；短边不足 1000 时邻域为 5，否则为 7；对墨迹及相邻边缘减弱滤波 |
| 纸面光照不均 | 分块估计纸面亮度并平滑插值；局部提亮最多 1.25 倍，笔画两侧使用连续增益以避免光晕 |
| 有可分辨但对比度偏低的墨迹 | 仅在 LAB 亮度通道做温和对比度拉伸，单次亮度调整最多 8/255，保持色度通道 |

噪声使用局部平坦区域的高通残差 MAD 估计；墨迹掩膜按纸面亮度差和噪声水平构造，
边缘羽化后与原图融合。处理后的原墨迹平均亮度变化若超过 6%，或墨迹保留比例低于 97%，
会放弃该候选并返回原图。诊断和处理会临时归一化黑底白字拓印，返回时恢复其极性，
以便后续反色检测仍能正确记录拓印信息。输出保留摆正方向后的尺寸，不缩小原图。

```python
from PIL import Image
from zhimo.vision.denoise import denoise_image

with Image.open("image_test/test.png") as image:
    processed, info = denoise_image(image)  # 默认 adaptive
    basic, basic_info = denoise_image(image, mode="basic")

print(info["applied"], info["reason"], info["steps"])
print(info["quality_before"], info["quality_after"])
processed.save("denoised.png")
```

`basic` 保留旧的“3×3 中值 + 双边滤波 + LAB 亮度 CLAHE”像素处理，供对照实验使用；
网页开启去噪时使用 `adaptive`。新增诊断包含实际处理步骤、参数、处理前后指标以及笔画变化：

- `enabled` 表示调用了去噪；`applied` 表示滤波或亮度处理实际改变了像素，不包含 EXIF 方向调整。
- `reason` 包括 `no_actionable_degradation`、`image_too_small`、`degradation_detected`、`stroke_guard_reverted`、`basic_requested`。
- `quality_before` / `quality_after` 在同一归一化极性下测量；`noise_sigma`、`background_variation`、`ink_contrast` 为 0–255 灰度尺度，`impulse_ratio`、`ink_fraction` 为 0–1 比例；`laplacian_variance` 是拉普拉斯响应方差，不是噪声概率。
- `stroke_change` 为原墨迹区域平均灰度变化除以 255；`stroke_retention` 为原墨迹位置仍保留至少 70% 纸墨亮度差的比例，无墨迹时为 `null`。这些是保护性启发式，不能证明所有笔锋、飞白均未受影响。
- 若保护检查触发回退，`rejected_candidate` 记录被放弃的步骤、指标和笔画变化，前后主指标则对应实际返回的原图。
- 这些前后指标位于 HTTP 响应的 `denoise` 对象内；原有顶层 `quality` 仍描述后续识别实际使用的图片。

此版本不自动漂白均匀泛黄的纸张，不生成或补画笔画，也不专门修复严重模糊、JPEG 块效应或成团污渍。
噪声与纸纹、孤立墨点仍可能混淆；复杂彩色纸和密集作品的背景估计也可能不准确。
处理更平滑或模型置信度更高不代表作者识别更准确，仍需带作者真值的独立验证集评估。

无需训练或联网即可运行降噪回归和可复现对照：

```powershell
python -B -m unittest tests.unit.test_denoise -v
python evaluation/evaluate_denoise.py --image image_test/test.png --out denoise_comparison
# 可选：用已部署权重对各版本做推理，不进行训练、不调用 Agent API
python evaluation/evaluate_denoise.py --image image_test/test.png --out denoise_model_comparison --recognize
```

对照脚本固定随机种子（默认 42），生成高斯噪声、脉冲噪声、阴影和低对比度样例，
保存 `results.json` 和 `comparison.png`，比较原图、旧算法和自适应算法的 MSE、PSNR、诊断和耗时。
只在这个对照脚本中将参考图长边限制到 1024；生产去噪保持原尺寸。
PSNR 在像素完全一致时保存为 `null`，对应 MSE 为 0。可选识别结果仅用于观察预测变化，不能当作准确率。

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

网页中的“知识检索”开关默认关闭。这样在 Ollama 尚未启动、`bge-m3` 尚未下载或 Chroma 尚未初始化时，普通网页识别不会默认等待知识库；需要风格背景和作品信息时可以在页面中手动打开该开关。RAG 不可用时，普通图像识别仍可运行；Web 和 Agent 会返回结构化诊断。

这里的默认值分为两层：网页前端会显式提交 `rag=false`，而直接调用 `/api/analyze` 时如果省略 `rag`，服务端仍按当前接口默认值 `true` 处理。需要稳定跳过检索的 HTTP 调用应显式传入 `rag=false`。

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

PDF 分页和 Markdown 清理的专项回归测试：

```powershell
python -B -m unittest tests.unit.test_report_pagination -v
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

完整安装 `requirements.txt` 时会自动安装固定版本 `grad-cam==1.5.7`。如果使用了旧环境、只安装了部分依赖，或环境同步后仍提示缺少模块，再执行：

```powershell
python -m pip install grad-cam==1.5.7
```

普通识别不依赖 Grad-CAM。
