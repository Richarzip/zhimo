"""Audit calligrapher records with Baidu Baike public API."""

from __future__ import annotations

import ast
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "src" / "zhimo" / "knowledge" / "data.py"
AUDIT_PATH = ROOT / "research" / "baidu_knowledge_audit.json"
ASSET_DIR = ROOT / "frontend" / "static" / "assets" / "calligraphers"
ENDPOINT = "https://baike.baidu.com/api/openapi/BaikeLemmaCardApi"
USER_AGENT = "ZhimoKnowledgeAudit/2.0 (research; Baidu Baike public API)"

def clean(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<sup>.*?</sup>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()

def load_data() -> list[dict[str, str]]:
    source = DATA_PATH.read_text(encoding="utf-8-sig")
    return ast.literal_eval(source.split("=", 1)[1].strip())

def card(query: str) -> dict:
    response = requests.get(ENDPOINT, params={"scope": 103, "format": "json", "appid": 379020, "bk_key": query}, headers={"User-Agent": USER_AGENT}, timeout=(10, 30))
    response.raise_for_status()
    return response.json()

def fields(data: dict) -> dict[str, str]:
    result = {}
    for item in data.get("card", []) or []:
        key = clean(item.get("name"))
        value = clean(";".join(map(str, item.get("value") or [])))
        if key and value:
            result[key] = value
    return result

def works(text: str) -> list[str]:
    result = []
    left, right = chr(0x300a), chr(0x300b)
    for value in re.findall(f"{left}([^{right}]+){right}", text):
        value = value.strip().split(chr(0xff08), 1)[0].strip()
        if value and value not in result:
            result.append(value)
    if not result and text:
        separators = "".join(chr(x) for x in (0x3001, 0x2c, 0xff0c, 0x3b, 0xff1b))
        for value in re.split("[" + separators + "]", text):
            value = value.strip()
            if value and len(value) <= 30 and value not in result:
                result.append(value)
    return result


def is_calligraphy_work(title: str) -> bool:
    positive = [
        "\u5e16", "\u7891", "\u94ed", "\u7ecf", "\u8d4b", "\u5e8f", "\u4e66", "\u5377", "\u8f74", "\u8054",
        "\u8349\u4e66", "\u884c\u4e66", "\u6977\u4e66", "\u96b6\u4e66", "\u7bc6\u4e66",
        "\u5343\u5b57\u6587", "\u5170\u4ead", "\u77f3\u9f13\u6587", "\u8bf4\u6587", "\u5723\u6559",
    ]
    negative = [
        "\u6982\u8bba", "\u8bba\u7a3f", "\u5973\u79d1", "\u6587\u5b58", "\u753b", "\u56fe",
        "\u8bd7\u96c6", "\u4f5c\u54c1\u96c6", "\u5c0f\u8bf4", "\u8305\u5c4b", "\u74dc\u679c",
        "\u4e66\u53f2", "\u8bba\u4e66", "\u4e66\u6cd5\u9009", "\u89c2\u4e66", "\u4e1b\u7a3f",
    ]
    return any(key in title for key in positive) and not any(key in title for key in negative)

def image_for_work(work: str) -> dict | None:
    try:
        data = card(work)
    except requests.RequestException:
        return None
    title, image = clean(data.get("title")), clean(data.get("image"))
    if title != work or not image or not is_calligraphy_work(title):
        return None
    return {"work": work, "title": title, "source_url": data.get("url"), "download_url": image}

def save_image(item: dict, name: str, index: int) -> dict:
    safe = re.sub(r"[^\w\-]+", "_", f"{name}_{item['work']}_{index}", flags=re.UNICODE).strip("_")
    path = ASSET_DIR / f"{safe}.jpg"
    try:
        response = requests.get(item["download_url"], headers={"User-Agent": USER_AGENT, "Referer": "https://baike.baidu.com/"}, timeout=(10, 60))
        response.raise_for_status()
        if not response.headers.get("Content-Type", "").startswith("image/") or len(response.content) < 4096:
            raise ValueError("not a usable image")
        path.write_bytes(response.content)
        item["path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        item["url"] = f"/static/assets/calligraphers/{path.name}"
    except Exception as exc:
        item["download_error"] = f"{type(exc).__name__}: {exc}"
    return item

def make_text(name: str, info: dict[str, str], abstract: str, work_list: list[str]) -> str:
    parts = [f"{name}\u3002"]
    birth, death = info.get("\u51fa\u751f\u65e5\u671f"), info.get("\u901d\u4e16\u65e5\u671f")
    if birth or death:
        unknown = "\u4e0d\u8be6"
        parts.append(f"\u751f\u5352\uff1a{birth or unknown}\u2014{death or unknown}\u3002")
    era = info.get("\u6240\u5904\u65f6\u4ee3")
    if era:
        parts.append("\u65f6\u4ee3\uff1a" + era + "\u3002")
    identity = info.get("\u804c\u4e1a") or info.get("\u8eab\u4efd") or (abstract.split("\u3002", 1)[0] if abstract else "")
    if identity:
        parts.append("\u8eab\u4efd\u7b80\u4ecb\uff1a" + identity + "\u3002")
    if work_list:
        parts.append("\u767e\u5ea6\u767e\u79d1\u4e3b\u8981\u4f5c\u54c1\uff1a" + "\u3001".join(f"\u300a{x}\u300b" for x in work_list[:8]) + "\u3002")
    parts.append("\u98ce\u683c\u5224\u65ad\u4e0e\u4f5c\u54c1\u5f52\u5c5e\u9700\u4eba\u5de5\u590d\u6838\u3002")
    return "".join(parts)
def audit_one(item: dict) -> dict:
    name = item["name"]
    record = {"name": name, "source": ENDPOINT}
    try:
        data = card(name)
    except requests.RequestException as exc:
        record.update({"status": "request_failed", "error": f"{type(exc).__name__}: {exc}"})
        return record
    title, info = clean(data.get("title")), fields(data)
    abstract = clean(data.get("abstract"))
    source_work_text = info.get("\u4e3b\u8981\u4f5c\u54c1") or info.get("\u4ee3\u8868\u4f5c\u54c1") or info.get("\u4f5c\u54c1")
    work_list = works(source_work_text) if source_work_text else works(item.get("text", ""))
    record.update({"status": "exact" if title == name else "redirect_or_fuzzy", "baidu_title": title, "url": data.get("url"), "abstract": abstract, "fields": info, "candidate_works": work_list, "images": []})
    if title == name:
        for work in work_list[:4]:
            found = image_for_work(work)
            if found:
                record["images"].append(save_image(found, name, len(record["images"]) + 1))
                if len(record["images"]) >= 2:
                    break
    record["verified_text"] = make_text(name, info, abstract, work_list)
    record["review_notes"] = ["????????????????????", "????????????????", "????????????????????????????"]
    return record

def write_data(records: list[dict]) -> None:
    lines = ["CALLIGRAPHER_KNOWLEDGE = ["]
    for record in records:
        text = record.get("verified_text") or f"{record['name']}????????????????"
        lines.extend(["    {", f"        \"name\": {record['name']!r},", f"        \"text\": {text!r},", "    },"])
    lines.append("]")
    DATA_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    audit = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "primary_source": ENDPOINT, "baidu_search_status": "web_and_image_search_blocked_by_captcha_or_antiFlag", "image_policy": "work_entry_only", "entries": []}
    for index, item in enumerate(load_data(), 1):
        result = audit_one(item)
        audit["entries"].append(result)
        print(f"[{index}] {result['name']}: {result['status']} images={len(result.get('images', []))}")
        time.sleep(0.15)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_data(audit["entries"])

if __name__ == "__main__":
    main()
