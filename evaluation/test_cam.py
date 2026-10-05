import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhimo.vision.recognizer import CalligrapherRecognizer
from PIL import Image
import base64

recognizer = CalligrapherRecognizer()
img = Image.open("./image_test/test.png")

result = recognizer.predict_with_cam(img)
print(f"书法家: {result['calligrapher']}")
print(f"置信度: {result['confidence']}")
print(f"关注说明: {result['evidence'].get('attention_note')}")

# 保存热力图到本地
if "heatmap" in result["evidence"]:
    with open("./image_test/cam_output.png", "wb") as f:
        f.write(base64.b64decode(result["evidence"]["heatmap"]))
    print("热力图已保存到 cam_output.png")
