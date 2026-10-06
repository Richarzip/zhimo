"""Audit calligrapher records with Baidu Baike public API."""

from __future__ import annotations

import ast
import html
import json
import os
import re
import tempfile
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

# These are target identities, not facts inferred from the current API response.
# Broad birth windows tolerate disputed dates while separating historical and
# modern namesakes. Missing evidence always requires review; see IDENTITY_AUDIT.md.
IDENTITY_GROUPS = (
    (("钟繇",), ("东汉", "汉末", "三国", "曹魏"), (100, 220)),
    (("王羲之", "王献之"), ("东晋", "晋朝", "晋代"), (250, 420)),
    (("智永",), ("南北朝", "陈朝", "隋"), (400, 618)),
    (("虞世南", "欧阳询", "褚遂良"), ("南北朝", "隋", "唐"), (500, 650)),
    (("孙过庭", "张旭", "颜真卿", "怀素", "柳公权", "李阳冰", "唐玄宗"), ("唐",), (600, 850)),
    (("苏轼", "黄庭坚", "米芾", "蔡襄", "宋徽宗"), ("北宋", "宋朝", "宋代"), (900, 1150)),
    (("宋高宗", "朱熹"), ("南宋", "宋朝", "宋代"), (1050, 1250)),
    (("赵孟頫", "鲜于枢", "康里巎巎"), ("元", "宋末"), (1150, 1350)),
    (("沈周", "文徵明", "祝允明", "唐寅", "陈道复", "王宠", "徐渭", "董其昌", "倪元璐", "张瑞图"), ("明",), (1400, 1650)),
    (("王铎", "傅山", "八大山人"), ("明", "清初"), (1500, 1700)),
    (("金农", "郑板桥", "刘墉", "伊秉绶", "邓石如", "成亲王"), ("清",), (1600, 1800)),
    (("赵之谦", "何绍基"), ("清",), (1750, 1900)),
    (("吴昌硕", "王福庵", "于右任", "弘一", "毛泽东", "鲁迅", "沙孟海", "启功", "林散之"),
     ("清", "民国", "近代", "现代", "近现代"), (1800, 1930)),
)
EXPECTED_IDENTITIES = {
    name: {"eras": eras, "birth_range": birth_range}
    for names, eras, birth_range in IDENTITY_GROUPS for name in names
}
# Additional independently known identifiers for especially ambiguous names.
for _name, _identifiers in {
    "张旭": ("伯高", "张长史"),
    "王羲之": ("逸少", "王右军"),
    "颜真卿": ("清臣", "颜鲁公", "颜平原"),
    "刘墉": ("崇如", "石庵"),
}.items():
    EXPECTED_IDENTITIES[_name]["identifiers"] = _identifiers


def verify_identity(name: str, title: str, info: dict[str, str], abstract: str) -> dict:
    """Require independent target-era and calligraphy evidence before publishing."""
    expected = EXPECTED_IDENTITIES.get(name)
    reasons = []
    if title != name:
        return {"status": "redirect_or_fuzzy", "verified": False,
                "reasons": ["词条标题与目标姓名不一致，需人工确认别名或消歧结果。"]}
    if expected is None:
        return {"status": "needs_manual_review", "verified": False,
                "reasons": ["目标人物尚无独立身份约束。"]}

    # Do not infer identity from candidate works or the existing knowledge text:
    # both may contain inherited contamination or references to another person.
    introduction = re.split(r"人物生平|主要成就|人物经历", abstract, maxsplit=1)[0][:300]
    profession = " ".join(info.get(key, "") for key in ("职业", "身份"))
    role_evidence = re.search(r"书法家|书画家|書法家|書畫家", profession + " " + introduction)
    if not role_evidence:
        reasons.append("缺少明确的书法家或书画家身份说明。")

    era = info.get("所处时代", "")
    birth = info.get("出生日期", "")
    years = [int(value) for value in re.findall(r"(?<!\d)(\d{3,4})(?!\d)", birth)]
    if not years:
        # Only the subject's opening date, never dates of admired artists or works.
        opening = re.match(re.escape(name) + r"\s*[（(]([^）)]*)[）)]", introduction)
        if opening:
            year = re.search(r"(?<!\d)(\d{3,4})(?!\d)", opening.group(1))
            if year:
                years = [int(year.group(1))]
    start, end = expected["birth_range"]
    date_conflict = bool(years and any(not start <= year <= end for year in years))
    era_conflict = bool(era and not any(token in era for token in expected["eras"]))
    if date_conflict:
        reasons.append(f"出生年份 {years} 与目标人物年代范围 {start}–{end} 不符。")
    if era_conflict:
        reasons.append(f"词条时代“{era}”与目标人物时代不符。")
    if not era and not years:
        reasons.append("缺少可核对的时代或出生年份；不能以师承、作品名中的朝代代替。")

    identifiers = expected.get("identifiers", ())
    identity_text = " ".join([introduction, *(info.get(key, "") for key in ("字", "别名", "号"))])
    if identifiers and not any(value in identity_text for value in identifiers):
        reasons.append("尚未核对目标人物的字、号或通称。")
    status = "identity_mismatch" if date_conflict or era_conflict else "needs_manual_review" if reasons else "identity_verified"
    return {"status": status, "verified": status == "identity_verified",
            "expected_eras": list(expected["eras"]), "expected_birth_range": [start, end],
            "reasons": reasons}

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
    identity = verify_identity(name, title, info, abstract)
    source_work_text = info.get("主要作品") or info.get("代表作品") or info.get("作品")
    work_list = works(source_work_text) if source_work_text else []
    record.update({
        "status": identity["status"], "identity_check": identity,
        "baidu_title": title, "url": data.get("url"), "abstract": abstract,
        "fields": info, "candidate_works": work_list, "images": [],
        "review_notes": ["姓名相同不等于人物相同；身份和时代须共同匹配。",
                         "人物身份核对不代表每条生平、作品归属和图片版权已经人工审核。"],
    })
    if not identity["verified"]:
        record["review_notes"].extend(identity["reasons"])
        return record
    for work in work_list[:4]:
        found = image_for_work(work)
        if found:
            record["images"].append(save_image(found, name, len(record["images"]) + 1))
            if len(record["images"]) >= 2:
                break
    record["verified_text"] = make_text(name, info, abstract, work_list)
    return record

def write_data(records: list[dict]) -> None:
    existing = load_data() if DATA_PATH.exists() else []
    merged = {item["name"]: dict(item) for item in existing}
    for record in records:
        text = record.get("verified_text")
        if record.get("status") != "identity_verified" or not isinstance(text, str) or not text.strip():
            continue
        identity = verify_identity(
            record.get("name", ""), record.get("baidu_title", ""),
            record.get("fields") or {}, record.get("abstract", ""),
        )
        if not identity["verified"]:
            continue
        merged[record["name"]] = {"name": record["name"], "text": text}

    lines = ["CALLIGRAPHER_KNOWLEDGE = ["]
    for item in merged.values():
        lines.extend(["    {", f"        \"name\": {item['name']!r},", f"        \"text\": {item['text']!r},", "    },"])
    lines.append("]")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=DATA_PATH.parent,
            prefix=f".{DATA_PATH.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write("\n".join(lines) + "\n")
        os.replace(temporary_path, DATA_PATH)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
