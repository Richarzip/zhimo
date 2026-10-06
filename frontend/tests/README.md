# 前端回归测试

在项目目录（包含 `frontend/` 的目录）运行：

```bash
node --test frontend/tests/test_app.cjs
python -B -m unittest discover -s frontend/tests -p 'test_*.py' -v
```

JavaScript 测试使用 Node.js 自带的测试工具，不需要 npm 依赖。Python 测试需要 `aiohttp`、`Pillow`、`numpy`，无需模型权重、PyTorch 或运行中的 Ollama。

- `test_app.cjs`：执行页面原始脚本，模拟 DOM、图片解码和网络，覆盖换图/重置/乱序响应、重复提交、HTTP 与业务错误、上传限制、质量空值、加载气泡收尾、旧证据清理、历史图片 URL 生命周期和单/多字模式选项。
- `test_recognizer.py`：使用原推理方法和模拟张量/Grad-CAM，检查 TTA 四视图平均、热力图类别对齐与 CAM 失败回退。
- `test_server.py`：覆盖实际本地 HTTP 接口、上传校验、服务繁忙/取消后的并发限制，以及 RAG/CAM/TTA 和分析模式逐轮传入聊天管线；模型和知识库以测试替身替代。

这些测试不验证真实模型识别准确率；浏览器键盘操作仍需在实际页面检查。
