**智墨工作区代码审查记录 — 2026-10-06**

审查基线：`2639d03`。覆盖 Web 入口、前端交互、应用管线、Agent 工具、视觉推理/切分、知识库、PDF、训练、评估和资料审计脚本。首次检查开始时 Git 工作区干净；首次审查仅新增本报告。下列问题描述及文末测试表保留首次审查基线，后续修复状态记录如下。

**第 1–5 项修复状态（2026-10-06）**

| 项目 | 修复结果 | 回归覆盖 |
| --- | --- | --- |
| 1. 纯文字聊天 | 聊天允许无图，支持表单请求；空聊天、单/多字模式缺图、显式无效图片仍拒绝；无图可导出文字 PDF。 | 真实 HTTP 穿过聊天管线，连续两轮保持会话；上传边界、文本 PDF。 |
| 2. 摘要后回答丢失 | 使用当前消息 ID 和原历史消息 ID 区分本轮输出，兼容历史缩短及当前用户消息被摘要移除。 | 回答与工具结果保留，旧识别结果不混入本轮。 |
| 3. Agent 权重配置 | 统一使用应用工厂，按权重配置缓存；兼容单模型无 ensemble，提示词按实际模型说明。 | 默认、自定义单/多权重、切换配置、不同工作目录、三种识别工具。 |
| 4. 审计失败覆盖知识 | 仅发布 exact 且非空的核验文本；其余保留旧记录，文件通过临时文件原子替换。 | 网络失败、空文本、缺失/模糊状态、部分成功、替换失败。 |
| 5. 覆盖历史最佳模型 | 分别保存历史 best_acc 与本轮 val_acc；兼容旧断点时参考输出目录中模型/标签一致的最佳文件。 | 历史 0.90、旧断点 0.80、续训 0.85 不覆盖，0.95 才更新；并列分数不覆盖。 |

第 1–5 项完成时验证：`tests/` 29 项、`frontend/tests/` Python 测试 27 项，合计 **56 项通过**。同时修正了已有测试的路径分隔符可移植性和 OpenCV 模块隔离问题。检查点另通过真实 PyTorch 保存/加载验证。未运行完整训练或真实外部模型服务；该轮未处理第 6–24 项。

回归测试：

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s frontend/tests -p 'test_*.py' -v
```

**第 6–15 项修复状态（2026-10-06）**

| 项目 | 修复结果 | 回归覆盖 |
| --- | --- | --- |
| 6. 分析开关与模式 | 每轮完整传入 Agent 状态；RAG 关闭时禁止访问知识库，CAM 关闭走普通识别，TTA 传至所有识别工具。新增默认自动模式，显式单字/多字由提示词和工具共同约束。 | HTTP 到聊天管线、真实 LangChain 图与 ToolNode 注入、同会话反向切换开关、逐字 TTA、旧接口直接检索兼容、前端模式与 CAM 偏好。 |
| 7. CAM 空间布局 | Swin 转换 NHWC，ViT 系列移除前缀 token 并恢复空间网格，ResNet/EfficientNet 选择池化前层；集成复用代表成员的转换。 | 随机初始化的真实 ConvNeXt、Swin、ResNet、EfficientNet、ViT 模型及 Swin 集成成员；检查原始空间形状。 |
| 8. CAM 坐标对齐 | 从实际 CAM 输入张量反归一化底图，共用标准化常量，避免整图缩放与中心裁剪错位。 | 单模型和集成叠图逐像素对照输入中心裁剪，保留既有 TTA 类别与失败回退测试。 |
| 9. 切分回退 | 2–30 只作为优选范围；回退保留有效多字候选，再考虑单个非空候选。 | 真实 OpenCV 的 25/36/49 字网格、长作品、空白、单区域和合并连通域。 |
| 10. PDF 裁切 | 根据字形像素宽度换行，所有文字逐行分页，图片和 CAM 连同标题预留空间，长示例说明可跨页。 | 实际生成 PDF，核验长正文/字段/知识/URL 的完整文本、每次绘制边界、图片位置和页数。 |
| 11. 加载气泡 | 按 requestId 原位结束 loading，涵盖成功、业务/网络失败和取消。 | 原生 Node 执行页面脚本，核验消息数量、loading 收尾和乱序返回。 |
| 12. 旧结果残留 | 新请求与每次结果渲染先清理结果面板及下载事件，再显示本轮内容。 | 识别结果后接纯文字、异常，旧热力图/切分/诊断/参考图/PDF 均清理。 |
| 13. 历史图片失效 | Blob URL 在历史仍引用时保留，清空历史后释放；保留当前预览、去重释放。 | 换图后历史重绘、多轮复用图片、新对话与全重置的 URL 生命周期。 |
| 14. 重置保留隐藏记忆 | 全重置轮换 sessionId，并取消请求、隔离迟到响应。 | 重置后的请求使用新会话，旧响应不能污染新会话。 |
| 15. Chroma 路径忽略配置 | 构建器使用当前统一配置，相对路径基于项目根目录解析为绝对路径；Agent 缓存按目录更新。 | 默认/相对/绝对/用户主目录路径、改变工作目录、运行时切换配置；不写实际数据库。 |

第 6–15 项完成时验证：`tests/` **51 项**、`frontend/tests/` Python **27 项**、原生 Node **51 项**，合计 **129 项通过**，没有跳过可选集成测试。`git diff --check` 通过。第 1–5 项回归包含在内；该轮未处理第 16–24 项。

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s frontend/tests -p 'test_*.py' -v
node --test frontend/tests/test_app.cjs
```

本次真实集成测试使用 LangChain 1.4.3、timm 1.0.30、grad-cam 1.5.7；依赖仅补充到 `/tmp` 隔离目录，执行 Python 为 `/home/hy/miniconda3/envs/record/bin/python`，Node 为 `/tmp/zhimo-node-test/nodejs_wheel/bin/node`。Python 完整测试使用 `PYTHONPATH=/tmp/zhimo-agent-review-deps:/tmp/zhimo-vision-review-deps:src`。缺少 Agent/CAM 可选依赖时对应集成测试会跳过。没有访问真实 DeepSeek/Ollama 服务，没有改变现有知识库或模型权重，也未验证训练权重准确率、完整训练及真实浏览器端到端交互。

**第 16–24 项修复状态（2026-10-06）**

| 项目 | 修复结果 | 回归覆盖 |
| --- | --- | --- |
| 16. 同名人物错采 | 独立时代/出生范围与书法身份共同核验，部分易混淆人物再核字/号；模糊、缺证据或冲突结果不能自动发布，写入时再次核验。张旭源码定点纠正；旧审计离线重分类并隔离错误文图。 | 现代科研人员、现代同名书法家、日期/时代冲突、缺字段、模糊标题、沈周、写入复查；第4项合并/原子写入仍通过。 |
| 17. 续训状态丢失 | 正确恢复模型、scheduler、optimizer，保存/恢复 RNG、早停和曲线等状态。缺少关键优化状态明确报错；旧文件缺少随机/历史状态提示局限。CPU回退不因保存RNG而重新初始化CUDA。 | 真实 AdamW + 预热/余弦调度的保存/恢复；下一步 loss、参数、动量、学习率、scheduler、随机序列一致；weights_only=True兼容，真实训练入口早停续训。 |
| 18. 单样本尾批 | 只丢弃恰好1张的训练尾批，保留其他不足batch的尾批和全部验证样本；batch_size<2、训练集<2或空验证集明确拒绝。 | 65/66/64/2张数据、0/1张训练集、batch_size=1；实际 BatchNorm 前后向不再崩溃。 |
| 19. max_samples 类型 | CLI按正整数解析；数据集入口也校验类型与范围。 | 默认不限、100整数、0/负数/非整数拒绝；临时真实图片集按解析后的参数限量。 |
| 20. 指标缺少类别 | classification_report指定完整标签及名称；混淆矩阵保持固定顺序并安全处理缺失行和全空输入。 | 只出现1类仍输出54类报告；54×54矩阵无NaN；实际validate与绘图入口。 |
| 21. 训练依赖缺失 | requirements补充matplotlib、seaborn、scikit-learn；pyproject新增包含9项直接依赖的train组；README提供安装方式。 | 临时目录实际构建wheel核验Requires-Dist，两种声明均完整；训练入口 --help 正常。 |
| 22. 泛黄方向相反 | 减U、加V并保留适量亮度衰减，零强度直接保留像素。 | 灰阶纸张由中性变暖黄，黑墨保持明显更暗；强度递增和输入不被原地修改。 |
| 23. 零裁切抹全图 | 确定选中边后检查该方向实际裁切像素数，零值不执行负零切片。 | 四边、零强度、小图、非方形单轴为零；有效裁切只修改选中边。 |
| 24. 评估固定Windows路径 | 新增 --data-root，默认项目dataset/test，目录/抽样先校验；兼容权重内作者ID，未知类别或两模型标签错序拒绝评估，报告记录实际路径。 | 临时中文/作者ID目录、可复现采样、绝对/相对路径、旧n/seed/out参数、错误输入不加载模型；替身模型完成完整报告。 |

当前验证：`tests/` **93 项**、`frontend/tests/` Python **27 项**、原生 Node **51 项**，合计 **171 项通过**，无可选集成测试跳过；`git diff --check` 通过。此前1–15项的回归包含在内，编号1–24的修复均已完成。

全量测试使用上轮隔离依赖，加上 `/tmp/zhimo-training-review-deps` 中的 seaborn 0.13.2：

```bash
PYTHONPATH=/tmp/zhimo-training-review-deps:/tmp/zhimo-agent-review-deps:/tmp/zhimo-vision-review-deps:src \
MPLCONFIGDIR=/tmp/zhimo-matplotlib-review \
/home/hy/miniconda3/envs/record/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
/home/hy/miniconda3/envs/record/bin/python -B -m unittest discover -s frontend/tests -p 'test_*.py' -v
/tmp/zhimo-node-test/nodejs_wheel/bin/node --test frontend/tests/test_app.cjs
```

知识消歧策略、张旭更正的一手来源及53条旧审计的处置见 [IDENTITY_AUDIT.md](/home/hy/zhimo/zhimo/research/IDENTITY_AUDIT.md)：38条身份符合约束、14条待人工核对、1条身份冲突。未把待核对人物都判为错误；现有Chroma未被重建或改写。训练验证使用真实组件与微型输入、评估使用替身模型，没有运行完整训练或用真实训练权重评估准确率。

以下记录 24 项代码、数据或配置问题。P1 表示应优先处理的主流程中断或已有数据被覆盖问题；P2 表示影响结果正确性、配置或特定功能的问题。每项注明触发条件与验证依据，测试自身的问题另列。

**1. [P1] 页面允许纯文字提问，后端却无条件要求图片**

位置：[frontend/server.py:283](/home/hy/zhimo/zhimo/frontend/server.py:283)，[frontend/static/app.js:188](/home/hy/zhimo/zhimo/frontend/static/app.js:188)。

未选择图片、只输入问题时，前端允许发送 `mode=chat`，但 `read_upload()` 在区分模式前就检查图片是否存在。真实 HTTP 请求包含 `mode=chat`、`text` 和 `session_id`，返回 `400 {"error":"missing_image","message":"请上传图片。"}`。后续 `analyze_image()` 和 `run_chat()` 也假定图片一定存在，不能只放宽上传校验。应让聊天入口支持无图请求，并从已有会话按需取得图片。

**2. [P1] 对话历史被摘要压缩后，本轮回答可能变成空字符串**

位置：[pipeline.py:157](/home/hy/zhimo/zhimo/src/zhimo/application/pipeline.py:157)，[pipeline.py:176](/home/hy/zhimo/zhimo/src/zhimo/application/pipeline.py:176)，[builder_impl.py:97](/home/hy/zhimo/zhimo/src/zhimo/agent/builder_impl.py:97)。

代码先保存旧历史长度 `before`，调用 Agent 后通过 `messages[before:]` 取新消息；但启用的摘要中间件会删除旧消息并缩短历史。用 Agent 替身复现：调用前 30 条、摘要后连同正确回答共 21 条，接口得到 `reply=""`，工具结果也丢失。应按消息身份或当前轮执行事件确定输出，不能假定消息数组只增长。

**3. [P1] Agent 忽略自定义权重配置，健康检查与实际推理使用不同路径**

位置：[tools_impl.py:31](/home/hy/zhimo/zhimo/src/zhimo/agent/tools_impl.py:31)，[ensemble_impl.py:35](/home/hy/zhimo/zhimo/src/zhimo/vision/ensemble_impl.py:35)。

Web 工厂读取 `ZHIMO_MODEL_PATH` / `ZHIMO_MODEL_PATHS`，Agent 的所有识别工具却调用无参数 `EnsembleRecognizer()`，仍找当前工作目录下的两个默认文件。通过真实工厂函数配合构造器替身验证：设置 `/tmp/custom-model.pth` 后，Web 工厂收到该路径，Agent 仍收到 `./checkpoints/convnext.pth` 和 `./checkpoints/swin.pth`。仅有自定义权重时，页面健康检查可显示就绪，聊天识别仍报缺权重；从其他目录启动也会触发。应统一使用配置工厂，并兼容单模型返回结构。

**4. [P1] 知识审计请求失败时，会用问号覆盖原有知识文本**

位置：[baidu_knowledge_audit.py:143](/home/hy/zhimo/zhimo/research/baidu_knowledge_audit.py:143)，调用点 [157](/home/hy/zhimo/zhimo/research/baidu_knowledge_audit.py:157)。

网络异常时 `audit_one()` 返回 `request_failed`，`main()` 仍将所有记录交给 `write_data()` 覆盖 `data.py`。在临时目录中模拟请求异常，原有有效文本实际变成 `王羲之????????????????`。应保留失败记录的旧文本，只更新核验成功的记录，并采用原子写入。复现没有改动项目知识数据。

**5. [P1] 从普通 epoch 检查点恢复后，较差模型会覆盖历史最佳文件**

位置：[train.py:499](/home/hy/zhimo/zhimo/training/train.py:499)，恢复点 [453](/home/hy/zhimo/zhimo/training/train.py:453)，覆盖点 [510](/home/hy/zhimo/zhimo/training/train.py:510)。

检查点的 `best_acc` 实际保存本轮 `val_acc`。例如历史最佳为 0.90，恢复的某轮分数为 0.80，下一轮得到 0.85，就会满足“超过最佳”的判断，把原先 0.90 的最佳权重覆盖。该结论由保存、恢复和比较分支直接确定，未运行完整训练。应分别保存本轮准确率和历史最佳值，恢复时保留历史最佳。

**6. [P2] 页面上的 RAG、Grad-CAM、TTA 开关不影响聊天分析**

位置：[app.js:204](/home/hy/zhimo/zhimo/frontend/static/app.js:204)，[pipeline.py:155](/home/hy/zhimo/zhimo/src/zhimo/application/pipeline.py:155)，[tools_impl.py:134](/home/hy/zhimo/zhimo/src/zhimo/agent/tools_impl.py:134)。

页面固定发送 `mode=chat`，而 `run_chat()` 不读取这三个开关。Agent 提示词要求检索知识，质量分析工具固定调用 CAM，识别工具也没有传递 TTA。替身实验中，两组相反开关 `(rag=true,cam=true,tta=false)` 和 `(false,false,true)` 传给 `Agent.invoke()` 的输入、配置完全相同。应将选项传入工具运行上下文并执行约束；模式按钮同样没有影响实际请求。

**7. [P2] CAM 没有适配各 backbone 的特征布局，Swin 会静默输出错误热力图**

位置：[recognizer_impl.py:160](/home/hy/zhimo/zhimo/src/zhimo/vision/recognizer_impl.py:160)，[234](/home/hy/zhimo/zhimo/src/zhimo/vision/recognizer_impl.py:234)，[ensemble_impl.py:213](/home/hy/zhimo/zhimo/src/zhimo/vision/ensemble_impl.py:213)。

使用真实 PyTorch、`timm==1.0.30` 和 `grad-cam==1.5.7`，以随机初始化模型验证：ConvNeXt 特征 `(1,768,7,7)` 得到正常的 `(1,7,7)` CAM；Swin 特征 `(1,7,7,768)` 被误按 NCHW 处理，生成 `(1,7,768)` CAM 后硬缩成 224×224，把通道轴当成空间轴。单独用 Swin 或由 Swin 解释集成结论时触发。

同一适配缺陷还影响代码明确支持的其他 backbone：ResNet 选择整个已池化 encoder，得到 `(1,512)`；ViT 得到 `(1,197,192)`，两者实测均报 `Invalid grads shape`。应选择池化前的空间层，并为 Swin/ViT 提供适合的特征转换。维度规则可参见 [GradCAM 实现](https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/pytorch_grad_cam/grad_cam.py)。

**8. [P2] CAM 叠图坐标与模型实际输入不一致**

位置：[recognizer_impl.py:238](/home/hy/zhimo/zhimo/src/zhimo/vision/recognizer_impl.py:238)，[ensemble_impl.py:217](/home/hy/zhimo/zhimo/src/zhimo/vision/ensemble_impl.py:217)。

模型先把图片缩放为 256×256，再中心裁剪为 224×224；叠图底图却把整图直接缩为 224×224。真实 torchvision 复现：原图 `[32:48,32:48]` 黑块在模型输入中位于 `(16,16)-(31,31)`，叠图底图中却位于 `(28,28)-(41,41)`。热力图会指向错误笔画。应统一几何预处理，或将 CAM 正确映射回原图。

**9. [P2] 分割回退会用空连通域结果覆盖有效投影结果**

位置：[segmentation_impl.py:140](/home/hy/zhimo/zhimo/src/zhimo/vision/segmentation_impl.py:140)。

前一个 `elif` 不满足后，回退分支再次检查相同的 `2 <= len(boxes_proj) <= 30`，这条分支不可达。真实 OpenCV 实验：20×20 黑块、间隔 8 像素，5×5 网格返回 25 个候选；6×6 网格得到 `cc_count=0, proj_count=36`，最终却返回 0 个；7×7 同样丢掉全部 49 个区域。上层随即提示没有字符。应避免空结果覆盖有效候选，并明确处理候选数超出启发式范围的情况。

**10. [P2] PDF 中较长的回答直接画到页面外，内容被静默裁掉**

位置：[report.py:84](/home/hy/zhimo/zhimo/src/zhimo/application/report.py:84)，绘制函数 [36](/home/hy/zhimo/zhimo/src/zhimo/application/report.py:36)。

文本换行只增加 y 坐标，没有通用分页逻辑；正文、结果字段和热力图都可能越过页面底部。Pillow 12.3.0 实测约 4000 字回答仍只产生一页 PDF，6 次文字绘制超出页面高度 2339，结果字段被画到 y=2485、2514。应在每行/每块内容绘制前检查空间并新建页面。

**11. [P2] 每次聊天完成后，“正在回复”的气泡仍永久闪动**

位置：[app.js:200](/home/hy/zhimo/zhimo/frontend/static/app.js:200)，[app.js:285](/home/hy/zhimo/zhimo/frontend/static/app.js:285)。

发送时追加 `loading:true`，成功或失败时再次追加消息，`pushAiMessage()` 从不替换旧占位。用 V8 执行真实 `app.js` 和测试 DOM 替身，完成一轮后消息仍为 `[user, ai-loading, ai-answer]`，页面保留一个 `chat-loading` 元素。应按请求 ID 更新或移除对应占位，覆盖成功、失败和取消路径。

**12. [P2] 后续回答会沿用上一轮热力图、诊断、参考图及 PDF**

位置：[app.js:434](/home/hy/zhimo/zhimo/frontend/static/app.js:434)，[app.js:478](/home/hy/zhimo/zhimo/frontend/static/app.js:478)。

`renderResult()` 只在有字段时显示内容，字段缺失时不清理旧视图。同一图片继续对话时也不执行 `clearResult()`。V8 实测：第一轮返回 CAM/PDF/诊断/参考图，第二轮只返回文字，四类旧内容仍可见，PDF 下载事件仍指向第一轮文件。应在每轮渲染时完整更新结果区，或明确标注证据所属轮次。

**13. [P2] 更换图片会使聊天历史中的旧图片失效**

位置：[app.js:107](/home/hy/zhimo/zhimo/frontend/static/app.js:107)，[app.js:281](/home/hy/zhimo/zhimo/frontend/static/app.js:281)。

`clearFile()` 立即撤销旧 Blob URL，但聊天消息仍保存它；`renderChat()` 整体重建 DOM 后，旧图片必须重新加载已经撤销的 URL。V8 实测确认撤销列表包含旧地址，而新渲染的历史 HTML 仍引用该地址。应按历史消息的生命周期保留地址，删除历史时再释放。

**14. [P2] 重置清空了可见聊天，但保留服务端会话记忆**

位置：[app.js:550](/home/hy/zhimo/zhimo/frontend/static/app.js:550)，对照 [resetChat:421](/home/hy/zhimo/zhimo/frontend/static/app.js:421)。

`resetAll()` 删除图片与聊天记录，却不更换 `sessionId`。V8 实测重置后可见消息数为 0，而 session ID 不变；下一次请求仍使用原来的 LangGraph thread，旧历史会继续影响用户已看不到上下文的新回答。应统一“重置”和会话记忆的语义。

**15. [P2] `ZHIMO_CHROMA_DIR` 只影响配置/健康检查，没有影响实际知识库**

位置：[chroma.py:13](/home/hy/zhimo/zhimo/src/zhimo/knowledge/chroma.py:13)，配置点 [settings.py:54](/home/hy/zhimo/zhimo/src/zhimo/config/settings.py:54)。

实际 Chroma 构造器始终使用相对路径 `./chroma_db`。设置自定义目录后，配置读取正确，但构造器替身收到的仍是默认相对路径。从其他目录启动时还可能加载另一套空库，尽管健康检查检查的是项目内的库。应从统一配置读取绝对持久化目录。

**16. [P2] 人物审计只校验姓名，已把现代同名人物写入古代书法家知识**

位置：[baidu_knowledge_audit.py:128](/home/hy/zhimo/zhimo/research/baidu_knowledge_audit.py:128)，实际错误数据 [data.py:36](/home/hy/zhimo/zhimo/src/zhimo/knowledge/data.py:36)。

标题等于姓名就被标为 `exact`，没有验证时代、身份或条目消歧。当前源码将“张旭”写成“1961年8月4日”“教育科研工作者”，同时保留《古诗四帖》等作品，混合了两个人的信息。已只读检查 SQLite：现有持久化库仍是旧的唐代张旭文本，因此不能声称当前旧库检索已返回错误人物；使用当前源码重新初始化时才会写入错误数据。应验证人物身份，模糊匹配不得自动发布到正式知识数据。

**17. [P2] 断点续训未恢复优化器状态，实际学习率也发生跳变**

位置：[train.py:449](/home/hy/zhimo/zhimo/training/train.py:449)。

检查点保存 `optimizer`，恢复时却只加载模型和 scheduler。真实 PyTorch 微型实验：保存时 lr 为 `7.76178585210434e-05`，按当前恢复流程得到 `8e-06`；AdamW 状态条目从 1 变为 0。恢复后的训练不等同于继续原训练。应按正确顺序恢复优化器、调度器和相关训练状态。

**18. [P2] 训练最后一批只有一个样本时会崩溃**

位置：[train.py:409](/home/hy/zhimo/zhimo/training/train.py:409)，模型头 [205](/home/hy/zhimo/zhimo/training/train.py:205)。

训练 DataLoader 保留尾批，而模型头含 `BatchNorm1d`。例如 65 张图片、batch size 64，第二批只有一张。真实 PyTorch 实测抛出 `Expected more than 1 value per channel when training`。应调整训练采样/尾批策略，并单独处理总样本不足和 batch size 为 1 的配置。

**19. [P2] `--max_samples` 解析为字符串，使用时立即报切片错误**

位置：[train.py:334](/home/hy/zhimo/zhimo/training/train.py:334)，使用点 [162](/home/hy/zhimo/zhimo/training/train.py:162)。

默认值为 `None` 的配置项被统一声明为 `type=str`，所以 `--max_samples 100` 得到字符串 `"100"`。执行原始数据集类、使用临时目录复现 `TypeError: slice indices must be integers or None or have an __index__ method`。应显式使用整数类型并校验范围。

**20. [P2] 验证数据未覆盖全部类别时，生成分类报告会中断训练**

位置：[train.py:257](/home/hy/zhimo/zhimo/training/train.py:257)。

`target_names` 总是包含 54 个作者，但没有传完整 `labels`，类别集合由实际真值与预测推导。用真实 scikit-learn 1.9.1 复现：只出现类别 0、传 54 个名称时，报 `Number of classes, 1, does not match size of target_names, 54`；显式传 `labels=range(54)` 后正常产生 54 类报告。部分作者缺数据或调试子集会触发。混淆矩阵也应保持同一标签顺序。

**21. [P2] 安装文档列出的依赖不足以启动训练脚本**

位置：[train.py:28](/home/hy/zhimo/zhimo/training/train.py:28)，[requirements.txt](/home/hy/zhimo/zhimo/requirements.txt)，[pyproject.toml](/home/hy/zhimo/zhimo/pyproject.toml)。

训练脚本无条件导入 `matplotlib`、`seaborn`、`sklearn`，但正确解码后的 requirements 没有这三项，项目元数据也没有声明依赖。干净环境仅按 README 安装不能保证训练入口可导入。应声明训练所需依赖或提供明确的训练依赖分组。requirements 的 UTF-16 编码本身未计为 bug。

**22. [P2] 泛黄增强实际把白纸变成蓝色**

位置：[adverse_aug.py:60](/home/hy/zhimo/zhimo/training/adverse_aug.py:60)。

手写 YUV 转换中，增加 U、减少 V 会增强蓝色并降低红色，与泛黄目标相反。调用真实函数，白色 RGB `[255,255,255]` 在 severity=1 时变成 `[182,214,255]`。这会使生成的退化数据不符合标注的退化类型。应修正色度方向，并用代表性色块检查增强结果。

**23. [P2] 裁切像素数为零时，下/右裁切会抹掉整张图片**

位置：[adverse_aug.py:119](/home/hy/zhimo/zhimo/training/adverse_aug.py:119)，右侧分支 [125](/home/hy/zhimo/zhimo/training/adverse_aug.py:125)。

`int(size * 0.18 * severity)` 可能为 0，Python 的 `out[-0:]` 就是整张图。10×10 图、默认 severity=0.5、bottom 分支已复现全部 100 个像素变成同一颜色；severity=0 也可触发。应在裁切量为零时直接返回原图，不能执行负零切片。

**24. [P2] 集成评估脚本固定使用开发者的 Windows 数据路径**

位置：[evaluate_ensemble.py:34](/home/hy/zhimo/zhimo/evaluation/evaluate_ensemble.py:34)，参数定义 [69](/home/hy/zhimo/zhimo/evaluation/evaluate_ensemble.py:69)。

路径固定为 `D:\experiment3\dataset_total\dataset1`，命令行只允许 n、seed、out，没有数据目录参数。在当前 Linux 工作区该目录不存在，即使准备了本地数据与模型，也无法通过配置运行评估，必须先改源码。应提供数据根目录参数，并在启动时检查目录。

**测试、文档和验证范围**

| 检查 | 结果与限制 |
| --- | --- |
| `tests/unit` | 3 项中 2 通过、1 失败。测试硬编码 `a.pth;b.pth`，实现使用 `os.pathsep`；Linux 上解析成单一路径。README 本身说明使用系统分隔符，因此这是测试可移植性问题，未混算为生产缺陷。 |
| `frontend/tests/test_server.py` | 直接执行时有 1 项测试的 4 个子例因 OpenCV 重导入失败；预先 `import cv2` 后 14 项全通过。原因是 `patch.dict(sys.modules)` 清除了测试中首次导入的 cv2，属于测试隔离问题。 |
| `frontend/tests/test_recognizer.py` | 4 项通过，但替换了 GradCAM 和目标层，不能覆盖本报告发现的真实特征布局问题。 |
| 真实 HTTP | 已验证无图文字聊天返回 400；模型调用使用替身，没有访问外部模型服务。 |
| 真实视觉组件 | 使用实际 PyTorch/torchvision、匹配声明版本的 timm，以及随机初始化模型验证 CAM；实际 OpenCV 验证分割。没有验证训练权重的识别准确率。 |
| JavaScript | 本机无 Node，未执行 `node --test` 原测试套件；改用 V8 执行真实 app.js 和现有测试的 DOM 替身，定向复现第 11–14 项。不是实际浏览器端到端测试。 |
| PDF / 指标 / 训练片段 | 实际 Pillow 12.3.0、scikit-learn 1.9.1、PyTorch 微型输入；未执行完整训练。新增验证依赖仅装在 `/tmp`。 |
| 知识库 | SQLite 使用只读连接检查；模拟写入均重定向到临时目录。当前 SQLite 和知识源码已经不是相同文本版本。 |

README 另有失效说明：[第 89 行](/home/hy/zhimo/zhimo/README.md:89)等处称旧兼容入口仍存在，但根目录没有 `train.py`、`merge_split_data.py`、`calligrapher_recognizer.py`、`ensemble_recognizer.py`、`calligrapher_tool.py`、`rag_setup.py` 等文件。应修正文档命令，或补回承诺的兼容入口。

仓库没有提供实际模型权重、训练数据；本次也没有配置真实 LangChain/DeepSeek/Ollama 链路。因此不能据此保证真实模型准确率、外部服务行为或全部交互均正确。建议先修复第 1–5 项，再处理开关传递、视觉证据正确性、训练恢复和新增功能的回归覆盖。
