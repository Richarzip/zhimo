from dotenv import load_dotenv
load_dotenv()

import base64
from langchain.messages import HumanMessage

from agent_builder import create_calligraphy_agent

# 单次调用示例（无状态）；前端多轮对话见 frontend/server.py 的 mode=chat
agent = create_calligraphy_agent()

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