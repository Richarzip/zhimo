from dotenv import load_dotenv
load_dotenv()

import base64
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.prebuilt import InjectedState

from calligrapher_tool import identify_calligrapher,analyze_calligraphy,search_knowledge, analyze_multi_char

model = init_chat_model(
    "deepseek:deepseek-flash",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
)


# 3. 创建 Agent
agent = create_agent(
    model=model,
    tools=[identify_calligrapher, analyze_calligraphy, search_knowledge, analyze_multi_char],
    system_prompt="""你是一位书法鉴赏专家。

        工作流程：
        1. 用户上传图片后，先观察图片内容：
        - 如果是**单字**图片（只有一个字），调用 analyze_calligraphy 识别书法家。
        - 如果是**多字作品**（多个字或整幅作品），调用 analyze_multi_char 进行切分和综合识别。
        2. 拿到识别结果后，调用 search_knowledge 检索该书法家的风格知识。
        3. 综合识别结果和知识库内容，生成有依据的鉴赏回答。

        规则：
        1. 任何关于"这是谁的字"的判断，必须来自 analyze_calligraphy 或 analyze_multi_char。
        2. 不要根据图片内容自行猜测书法家。
        3. 解释风格特点时，必须引用 search_knowledge 返回的内容。
        4. 不要编造知识库中没有的信息。
        5. 根据工具返回的 reliability 或 consistency 调整回答语气：
        - high: 直接给结论
        - medium: 给结论但提醒"存在一定不确定性"
        - low: 不要强行下结论，建议用户提供更清晰的图片
        6. 解释判断原因时，必须引用:
           - analyze_calligraphy 返回的置信度和 Top-K 数据
           - evidence.attention_note（如果存在），说明"模型关注了哪些区域", 要针对这部分区域的笔画进行细致化的分析
        7. 如果 analyze_multi_char 返回的 note 中说明切分可能有误差，必须在回答中如实告知用户。
    """,
)

with open("./image_test/test.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

result = agent.invoke({
    "messages": [
        HumanMessage(content=[
            {"type": "text", "text": "这是谁的字？为什么？"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            }
        ])
    ]
})


for msg in result["messages"]:
    print(f"[{msg.type}] ", end="")
    if msg.type == "ai" and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"→ 调用 {tc['name']}")
    elif msg.type == "tool":
        print(f"工具返回：{msg.content[:200]}...")
    elif msg.content:
        print(f"{msg.content}")
    else:
        print()