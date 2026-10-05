"""Run a one-shot calligraphy Agent example from the project root."""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Zhimo calligraphy Agent once.")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(ROOT / "image_test" / "test.png"),
        help="Path to the image sent to the Agent.",
    )
    parser.add_argument(
        "--prompt",
        default="What calligrapher does this handwriting resemble, and why?",
        help="Question sent together with the image.",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    from langchain.messages import HumanMessage
    from zhimo.agent import create_calligraphy_agent

    load_dotenv(ROOT / ".env")
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with image_path.open("rb") as handle:
        image_b64 = base64.b64encode(handle.read()).decode("ascii")

    agent = create_calligraphy_agent()
    result = agent.invoke({
        "messages": [
            HumanMessage(content=[
                {"type": "text", "text": args.prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ])
        ]
    })

    for message in result.get("messages", []):
        print(f"[{message.type}] ", end="")
        if message.type == "ai" and message.tool_calls:
            print(" -> ".join(call.get("name", "tool") for call in message.tool_calls))
        elif message.type == "tool":
            print(f"{str(message.content)[:200]}...")
        elif message.content:
            print(message.content)
        else:
            print()


if __name__ == "__main__":
    main()