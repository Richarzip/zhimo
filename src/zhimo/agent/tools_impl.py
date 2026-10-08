import base64
from io import BytesIO
from typing import Annotated

from PIL import Image
from langchain.tools import tool
from langgraph.prebuilt import InjectedState

from ..application import factory as model_factory
from ..vision.preprocess_impl import detect_and_fix_inversion

import cv2
import numpy as np
from ..vision.quality_impl import assess_image_quality
from ..knowledge.chroma import build_vectorstore
from ..config import get_settings

# 复用当前权重配置的模型；配置改变后重新创建，避免沿用旧权重。
_recognizer = None
_recognizer_paths = None


def get_recognizer():
    global _recognizer, _recognizer_paths
    paths = tuple(model_factory.resolve_model_paths())
    if _recognizer is None or paths != _recognizer_paths:
        _recognizer = model_factory.get_recognizer()
        _recognizer_paths = paths
    return _recognizer


def _text_evidence(evidence: dict) -> dict:
    """只把文本化的可解释性信号给 LLM。

    evidence 里含 heatmap base64（几十 KB 的字符串）：文本模型读不了图，
    只会白占上下文 token；且热力图有时关注背景，展示在回复里是减分项。
    这里只保留 attention_note / cam_model / error 等文字字段。
    """
    return {
        key: evidence.get(key)
        for key in ("attention_note", "cam_model", "error")
        if evidence.get(key) is not None
    }


# 辅助函数：从 Agent 状态里提取图片 
def extract_image_from_state(state: dict) -> Image.Image | None:
    """从 Agent 的消息历史里找到最近一张图片（当前轮）。

    多轮对话时历史消息里可能有多张图，这里倒序取最新一条
    human 消息中的图片，保证识别的是用户当前上传的图。
    """
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if msg.type == "human" and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    url = block["image_url"]["url"]
                    if url.startswith("data:"):
                        b64 = url.split(",", 1)[1]
                        return Image.open(BytesIO(base64.b64decode(b64)))
    return None


def analysis_options(state: dict | None) -> dict:
    """Read request options injected by the graph, with CLI-compatible defaults."""
    return (state or {}).get("analysis_options") or {}


def _mode_error(state: dict, mode: str) -> dict | None:
    selected = analysis_options(state).get("analysis_mode", "auto")
    if selected not in ("auto", mode):
        tool_name = "analyze_multi_char" if selected == "multi" else "analyze_calligraphy"
        return {"error": f"本轮选择了 {selected} 模式，请调用 {tool_name}。"}
    return None


# 工具 1：书法家识别 
@tool
def identify_calligrapher(state: Annotated[dict, InjectedState]) -> dict:
    """识别书法作品的书法家身份。

    这是唯一能判断书法家身份的工具。
    不要根据图片内容自行猜测书法家。
    使用配置的识别模型；配置多个权重时进行软投票集成。

    返回：
        - calligrapher: 识别到的书法家姓名
        - confidence: 置信度（0-1）
        - top_k: Top-3 候选及置信度
        - all_probabilities: 所有类别概率
        - ensemble: 多模型集成详情，单模型时为 None
    """
    if error := _mode_error(state, "single"):
        return error
    options = analysis_options(state)
    image = extract_image_from_state(state)
    if image is None:
        return {"error": "没有找到图片，请上传书法作品"}

    # 反色预处理（处理拓印）：黑底白字 → 白底黑字
    image, inversion = detect_and_fix_inversion(image)

    recognizer = get_recognizer()
    result = recognizer.recognize(image, tta=options.get("tta", False))

    # 从 all_probabilities 里取 Top-3
    sorted_probs = sorted(
        result["all_probabilities"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_k = [{"name": name, "confidence": prob} for name, prob in sorted_probs[:3]]

    return {
        "calligrapher": result["calligrapher"],
        "confidence": result["confidence"],
        "top_k": top_k,
        "all_probabilities": result["all_probabilities"],
        "model_backbone": result["model_backbone"],
        "ensemble": result.get("ensemble"),
        "evidence": _text_evidence(result.get("evidence", {})),
        "inversion": inversion,
    }


# 工具 2：质量评估
@tool
def analyze_calligraphy(state: Annotated[dict, InjectedState]) -> dict:
    """分析书法作品，返回识别结果和图像质量报告。

    注意：本工具不会修复图像，而是评估图像质量并在回答中体现。
    如果图像质量差，识别结果会被标记为"低可信度"。

    返回：
        - calligrapher: 识别到的书法家姓名
        - confidence: 置信度（0~1）
        - top_k: Top-3 候选
        - quality: 图像质量报告（各畸变评分）
        - reliability: 综合可信度评级（high / medium / low）
        - ensemble: 多模型集成详情，单模型时为 None
    """
    if error := _mode_error(state, "single"):
        return error
    options = analysis_options(state)
    image = extract_image_from_state(state)
    if image is None:
        return {"error": "没有找到图片"}

    # 反色预处理（处理拓印）：黑底白字 → 白底黑字，
    # 质量评估和识别都基于校正后的图
    image, inversion = detect_and_fix_inversion(image)

    # 质量评估
    img_np = np.array(image.convert("RGB"))
    quality = assess_image_quality(img_np)

    # 开关由本轮状态注入；关闭 CAM 时不创建 GradCAM 或执行反向传播。
    recognizer = get_recognizer()
    predict = recognizer.predict_with_cam if options.get("cam", False) else recognizer.recognize
    result = predict(image, tta=options.get("tta", False))

    # 综合可信度
    q = quality["overall"]
    c = result["confidence"]
    if q < 0.3 and c > 0.7:
        reliability = "high"
    elif q < 0.5 or c > 0.5:
        reliability = "medium"
    else:
        reliability = "low"

    sorted_probs = sorted(
        result["all_probabilities"].items(),
        key=lambda x: x[1], reverse=True,
    )
    top_k = [{"name": n, "confidence": c} for n, c in sorted_probs[:3]]

    return {
        "calligrapher": result["calligrapher"],
        "confidence": result["confidence"],
        "top_k": top_k,
        "quality": quality,
        "reliability": reliability,
        "model_backbone": result["model_backbone"],
        "ensemble": result.get("ensemble"),
        "evidence": _text_evidence(result.get("evidence", {})),
        "inversion": inversion,
    }



# 全局向量库实例
_vectorstore = None
_vectorstore_path = None

def get_vectorstore():
    global _vectorstore, _vectorstore_path
    path = get_settings().chroma_dir
    if _vectorstore is None or path != _vectorstore_path:
        _vectorstore = build_vectorstore()
        _vectorstore_path = path
    return _vectorstore

# 书法家知识库检索工具
@tool
def search_knowledge(query: str, state: Annotated[dict | None, InjectedState] = None) -> str:
    """检索书法家风格知识库。

    当你需要解释某位书法家的风格特点、代表作品、
    或与其他书法家对比时，调用这个工具。

    Args:
        query: 检索查询，如 "褚遂良风格特点"、"颜真卿和柳公权的区别"

    Returns:
        检索到的知识文本，以【书法家名】标注
    """
    if not analysis_options(state).get("rag", True):
        return "本轮已关闭知识库检索；请依据已有图像证据回答，不声称进行了检索。"
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=2)

    if not docs:
        return "知识库中没有找到相关信息。"

    parts = []
    for doc in docs:
        name = doc.metadata.get("name", "未知")
        parts.append(f"【{name}】\n{doc.page_content}")

    return "\n\n".join(parts)



@tool
def analyze_multi_char(state: Annotated[dict, InjectedState]) -> dict:
    """分析多字书法作品：切分 + 逐字识别 + 综合判断。

    当用户上传的作品包含多个字（而非单字图片）时，调用这个工具。
    它内部会：
    1. 用传统方法切分图像，找出每个字的位置
    2. 对每个切出的字单独识别
    3. 综合所有识别结果，投票得出最终结论

    返回：
        - calligrapher: 综合判断的书法家姓名
        - confidence: 综合置信度
        - num_characters: 检测到的字数
        - per_char_results: 每个字的识别结果
        - consistency: 一致性评级（high/medium/low）
        - note: 说明（切分方法、异常情况等）
    """
    from ..vision.segmentation_impl import segment_auto
    
    if error := _mode_error(state, "multi"):
        return error
    options = analysis_options(state)
    image = extract_image_from_state(state)
    if image is None:
        return {"error": "没有找到图片"}

    # 反色预处理（处理拓印）：黑底白字 → 白底黑字
    image, inversion = detect_and_fix_inversion(image)
    img_np = np.array(image.convert("RGB"))

    # 1. 切分
    seg = segment_auto(img_np)
    # ToolMessage content is serialized before it reaches the chat pipeline.
    # Normalize coordinates here so custom/OpenCV segmenters cannot leak NumPy
    # scalar values into the LangChain response.
    boxes = [tuple(int(value) for value in box) for box in seg.get("boxes", [])]

    if not boxes:
        return {
            "error": "未能检测到字符",
            "note": "图像可能不是书法作品，或者切分失败",
        }

    # 2. 使用配置的模型逐字识别。
    recognizer = get_recognizer()
    per_char_results = []

    for (x, y, w, h) in boxes:
        # 裁出单字（加一点边距）
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_np.shape[1], x + w + pad)
        y2 = min(img_np.shape[0], y + h + pad)

        char_img = Image.fromarray(img_np[y1:y2, x1:x2])

        try:
            r = recognizer.recognize(char_img, tta=options.get("tta", False))
            per_char_results.append({
                "name": r["calligrapher"],
                "confidence": r["confidence"],
                "bbox": [int(x), int(y), int(w), int(h)],
            })
        except Exception as e:
            per_char_results.append({
                "name": "unknown",
                "confidence": 0.0,
                "error": str(e),
            })

    if not per_char_results:
        return {"error": "所有字符识别失败"}

    # 3. 综合投票（加权：置信度 × 票数）
    votes = {}
    for r in per_char_results:
        if r["name"] == "unknown":
            continue
        votes.setdefault(r["name"], []).append(r["confidence"])

    if not votes:
        return {"error": "无法得到有效识别结果"}

    # 每个候选的得分 = 票数 × 平均置信度
    scores = {
        name: len(confs) * (sum(confs) / len(confs))
        for name, confs in votes.items()
    }
    best_name = max(scores, key=scores.get)
    avg_conf = sum(votes[best_name]) / len(votes[best_name])

    # 一致性评级
    total_chars = sum(len(c) for c in votes.values())
    consistency_ratio = len(votes[best_name]) / total_chars
    if consistency_ratio > 0.7:
        consistency = "high"
    elif consistency_ratio > 0.4:
        consistency = "medium"
    else:
        consistency = "low"

    # 4. 说明
    notes = [f"使用 {seg['method']} 方法切分"]
    if consistency == "low":
        notes.append("各字识别结果分歧较大，结论不可靠")
    if len(boxes) < 3:
        notes.append("检测到的字符数量偏少，切分可能不完整")

    return {
        "calligrapher": best_name,
        "confidence": round(avg_conf, 4),
        "num_characters": len(boxes),
        "per_char_results": per_char_results,
        "consistency": consistency,
        "vote_distribution": {
            name: len(confs) for name, confs in votes.items()
        },
        "inversion": inversion,
        "note": "；".join(notes),
    }


# 工具 5：导出对话为 PDF（意图标记工具）
@tool
def export_pdf(state: Annotated[dict, InjectedState]) -> str:
    """将当前对话内容导出为 PDF 报告并自动下载。

    当用户要求"导出 PDF"、"生成报告"、"把对话保存为 PDF / 文件"、
    "把刚才的讨论整理成文档"等时调用本工具。

    本工具只负责确认导出意图：PDF 文件由前端根据对话记录生成并自动下载，
    不需要也不应返回文件内容或二进制数据。

    Returns:
        面向用户的导出确认信息。
    """
    return "好的，正在为您把当前对话记录导出为 PDF，文件生成后会自动下载。"
