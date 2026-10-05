# test_tool.py
from dotenv import load_dotenv
load_dotenv()

import base64
import sys
from pathlib import Path
from PIL import Image
from langchain.messages import HumanMessage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhimo.agent.tools import identify_calligrapher

# 读取测试图片
with open("test.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

# 构造一个模拟的 state
image = Image.open("test.jpg")
msg = HumanMessage(content=[
    {"type": "text", "text": "这是谁的字？"},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
])

# 直接调用工具（注意：注入参数需要手动传）
result = identify_calligrapher.invoke({"state": {"messages": [msg]}})
print(result)
