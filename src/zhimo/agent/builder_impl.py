"""Agent 工厂：统一构建书法鉴赏 Agent，支持多轮对话（MemorySaver checkpointer）。

agent.py 与 frontend/server.py 共用本模块，保证 system prompt 与工具集一致。
"""
from typing import NotRequired

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
    SummarizationMiddleware,
    dynamic_prompt,
)
from langgraph.checkpoint.memory import MemorySaver

from .tools_impl import (
    identify_calligrapher,
    analyze_calligraphy,
    search_knowledge,
    analyze_multi_char,
)

# 增强版系统提示词：要求结构化、有依据、内容丰富的鉴赏回答，并支持多轮对话
SYSTEM_PROMPT = """你是一位严谨的书法鉴赏专家，负责从图片证据出发分析书法作品。
所有关于"这是谁的字"的判断必须来自工具返回的识别结果，禁止自行猜测。

【工作流程】
1. 用户上传图片后，先观察是单字还是多字：
   - 单字图片：调用 analyze_calligraphy（内部包含图像质量评估、按配置进行模型识别，并在 CAM 开启时生成热力图）。
   - 多字作品：调用 analyze_multi_char（切分 + 逐字识别 + 综合投票）。
2. 拿到识别结果后，在本轮 RAG 开启时调用 search_knowledge 检索该书法家的风格知识。
3. 综合识别结果、可用的模型或逐字投票一致性、质量报告、热力图和知识库内容，生成丰富、有条理的回答。

【回答结构要求】
每条完整回答请尽量覆盖以下部分（证据不足时可灵活调整，但不要遗漏结论）：
1. 结论先行：识别到的书法家是谁，置信度多少，综合可靠性（high/medium/low）如何。
2. 判断依据：说明依据哪些证据——置信度、Top-K 候选的差距、Grad-CAM 热力图主要关注区域（attention_note，针对该区域笔画具体分析）、多字投票的一致性。
3. 模型依据：按工具返回的 model_backbone 和 ensemble 说明识别来源，不预设模型架构或数量。
   - ensemble 有值时：按 members 中实际返回的模型说明软投票集成；agreement 为 true 时说明各成员判断一致，为 false 时列出各成员的 calligrapher 与 confidence，如实解释分歧。
   - ensemble 为 None 时：结合 model_backbone 说明单模型来源，不得声称使用双模型或经过双模型一致性验证。
   - 工具未返回模型详情时：只说明已有识别结果和逐字投票证据，不推断模型数量或成员判断。
4. 风格特征：引用 search_knowledge 返回的内容介绍用笔、结体、章法等风格特点。
5. 图像质量说明：泛黄/褪色/噪声/模糊评分对判断的影响；若 inversion.was_inverted 为 true，说明"检测到黑底白字拓印，已自动反转为白底黑字再识别"。
6. 下一步引导：给出 1-2 条可操作的追问建议，例如"可以上传更清晰的单字图进一步确认"、"想对比某两位书法家也可以问我"。

【多轮对话规则】
1. 用户追问时，结合之前轮次的识别结果和已检索的知识回答，不要重复完整流程，除非用户明确要求重新分析。
2. 用户说"刚才那幅字"、"刚才的分析"等时，引用最近一次的识别结果。
3. 用户追问风格、代表作、与某位书法家的区别时，在本轮 RAG 开启时优先用 search_knowledge 检索后再回答。
4. 始终保持中文回答，语气专业、亲切、有温度。

【硬性规则】
1. 任何关于"这是谁的字"的判断必须来自 analyze_calligraphy 或 analyze_multi_char，禁止编造。
2. 解释风格特点时必须引用 search_knowledge 返回的内容，不得编造知识库中没有的信息。
3. reliability 为 low 或 consistency 为 low 时，如实说明不确定性，不强行下结论。
4. 引用 attention_note 时，要针对对应区域（如左上、中央）的笔画特征做具体分析，而不是复述区域名称。
5. 若 analyze_multi_char 的 note 说明切分可能有误差，必须在回答中如实告知用户。
6. 仅在 ensemble 有值时说明多模型判断，并如实转述成员与分歧；ensemble 为 None 或缺失时不得声称双模型一致。
7. 用 Markdown 组织回答：适当使用加粗、小标题、列表，让内容层次清晰，但不要过度堆砌。
"""


class CalligraphyState(AgentState):
    analysis_options: NotRequired[dict]


@dynamic_prompt
def request_prompt(request) -> str:
    options = request.state.get("analysis_options") or {}
    mode = options.get("analysis_mode", "auto")
    mode_rule = {
        "auto": "根据图片决定调用单字或多字分析工具。",
        "single": "用户选择单字分析，只使用 analyze_calligraphy / identify_calligrapher，不调用 analyze_multi_char。",
        "multi": "用户选择多字分析，只使用 analyze_multi_char，不调用单字分析工具。",
    }.get(mode, "根据图片选择分析工具。")
    rag_rule = ("RAG 已开启，按需检索知识库。" if options.get("rag", True) else
                "RAG 已关闭，不调用 search_knowledge，不声称本轮检索过知识库；证据不足的风格信息请明确说明。")
    cam_rule = ("CAM 已开启，单字分析可使用工具返回的热力图证据。" if options.get("cam", True) else
                "CAM 已关闭，不声称本轮生成了热力图，不根据缺失的 attention_note 推断关注区域。")
    return SYSTEM_PROMPT + "\n【本轮选项，优先于通用工作流程】\n" + "\n".join((mode_rule, rag_rule, cam_rule))


def _tool_error_handler(exc: Exception, request) -> str:
    """工具异常 → 对模型可见的友好错误消息，避免 Agent 崩溃。

    返回错误内容会被包装成 status="error" 的 ToolMessage 交给模型，
    让模型知道工具失败了而不是中断整个对话。
    """
    tool_name = request.tool.name if request.tool else request.tool_call.get("name", "tool")
    brief = str(exc)[:120] if str(exc) else type(exc).__name__
    return (
        f"工具 `{tool_name}` 调用失败（{type(exc).__name__}: {brief}）。"
        "请检查输入是否有效（如图片格式、检索词），修正后重试，或换一种问法。"
    )


def _build_middleware(model):
    """用 LangChain 原生中间件做容错与保护，不自己实现重试/错误处理。

    组合顺序：列表靠前的中间件在外层。
    - ToolErrorMiddleware（外层）：工具异常统一转错误消息，Agent 不崩溃；
    - ToolRetryMiddleware（内层）：工具失败重试 1 次，耗尽后抛出交给外层处理；
    - ModelRetryMiddleware：模型调用失败自动重试 2 次（指数退避），
      耗尽后返回错误消息让 Agent 继续，而不是中断；
    - ToolCallLimitMiddleware：单轮最多 10 次工具调用，防止 Agent 陷入循环；
    - SummarizationMiddleware：多轮对话历史超过 30 条消息时自动摘要旧消息，
      防止上下文超限（触发前保留最近 20 条）。
    """
    return [
        request_prompt,
        ToolErrorMiddleware(on_error=_tool_error_handler),
        ToolRetryMiddleware(max_retries=1, on_failure="error"),
        ModelRetryMiddleware(max_retries=2, on_failure="continue"),
        ToolCallLimitMiddleware(run_limit=10, exit_behavior="continue"),
        SummarizationMiddleware(
            model=model,
            trigger=[("messages", 30)],
            keep=("messages", 20),
        ),
    ]


def create_calligraphy_agent(checkpointer=None):
    """构建书法鉴赏 Agent。

    Args:
        checkpointer: langgraph 检查点（多轮记忆用 MemorySaver）；
                      不传则创建无状态 Agent（单次调用）。
    """
    model = init_chat_model(
        "deepseek:deepseek-flash",
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return create_agent(
        model=model,
        tools=[
            identify_calligrapher,
            analyze_calligraphy,
            search_knowledge,
            analyze_multi_char,
        ],
        system_prompt=SYSTEM_PROMPT,
        state_schema=CalligraphyState,
        checkpointer=checkpointer,
        middleware=_build_middleware(model),
    )
