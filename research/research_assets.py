"""Audit the calligrapher knowledge base and download public reference images.

Sources are Wikimedia Wikipedia and Wikimedia Commons. The generated JSON is
an audit trail, not a claim that every stylistic sentence has been proven.
Entries without an exact article match are marked for manual review.
"""

from __future__ import annotations

import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "frontend" / "static" / "assets" / "calligraphers"
AUDIT_PATH = ROOT / "research" / "knowledge_audit.json"
USER_AGENT = "ZhimoKnowledgeAudit/1.0 (local research; contact project owner)"


def knowledge() -> list[dict]:
    source = (ROOT / "src" / "zhimo" / "knowledge" / "data.py").read_text(encoding="utf-8-sig")
    return ast.literal_eval(source.split("=", 1)[1].strip())


def api(params: dict, *, endpoint: str = "https://zh.wikipedia.org/w/api.php") -> dict:
    response = requests.get(
        endpoint,
        params={**params, "format": "json", "utf8": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=(8, 20),
    )
    response.raise_for_status()
    return response.json()


def wiki_page(name: str) -> dict:
    params = {
        "action": "query",
        "titles": name,
        "prop": "extracts|info|pageimages",
        "exintro": 1,
        "explaintext": 1,
        "inprop": "url",
        "piprop": "original",
        "pithumbsize": 900,
    }
    try:
        data = api(params, endpoint="https://en.wikipedia.org/w/api.php")
    except requests.RequestException:
        try:
            data = api(params)
        except requests.RequestException as exc:
            return {"exact": False, "error": f"Wikipedia unavailable: {type(exc).__name__}"}
    pages = list(data.get("query", {}).get("pages", {}).values())
    page = pages[0] if pages else {}
    if "missing" in page:
        try:
            search = api({"action": "query", "list": "search", "srsearch": name, "srlimit": 3})
            return {"exact": False, "search": [x.get("title") for x in search.get("query", {}).get("search", [])]}
        except requests.RequestException:
            return {"exact": False, "search": []}
    return {
        "exact": True,
        "title": page.get("title"),
        "pageid": page.get("pageid"),
        "url": page.get("fullurl") or f"https://zh.wikipedia.org/wiki/{quote(page.get('title', name))}",
        "extract": page.get("extract", ""),
        "thumbnail": (page.get("original") or {}).get("source"),
    }


def commons_images(name: str, limit: int = 2) -> list[dict]:
    try:
        data = api({
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{name} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": 8,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": 1200,
        }, endpoint="https://commons.wikimedia.org/w/api.php")
    except requests.RequestException:
        return []
    results = []
    for page in data.get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        mime = info.get("mime", "")
        url = info.get("thumburl") or info.get("url")
        if not url or not mime.startswith("image/"):
            continue
        metadata = info.get("extmetadata") or {}
        results.append({
            "title": page.get("title", ""),
            "source_url": f"https://commons.wikimedia.org/wiki/{quote(page.get('title', ''))}",
            "download_url": url,
            "mime": mime,
            "artist": (metadata.get("Artist") or {}).get("value", ""),
            "license": (metadata.get("LicenseShortName") or {}).get("value", ""),
        })
        if len(results) >= limit:
            break
    return results


def safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_") or "unknown"


def download(url: str, path: Path) -> None:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def audit_entry(item: dict) -> dict:
    name = item["name"]
    wiki = wiki_page(name)
    images = commons_images(name)
    saved = []
    for image_index, image in enumerate(images, start=1):
        extension = ".jpg" if "jpeg" in image.get("mime", "") else ".png"
        path = ASSET_DIR / f"{safe_name(name)}_{image_index}{extension}"
        try:
            download(image["download_url"], path)
            image["path"] = str(path.relative_to(ROOT)).replace("\\", "/")
            image["url"] = f"/static/assets/calligraphers/{path.name}"
            saved.append(image)
        except Exception as exc:
            image["download_error"] = f"{type(exc).__name__}: {exc}"
    text = item.get("text", "")
    dates = re.search(r"（([^）]{3,20})）", text)
    return {
        "name": name,
        "knowledge_dates": dates.group(1) if dates else None,
        "wiki": wiki,
        "images": saved,
        "review_status": "source_found" if wiki.get("exact") else "needs_manual_review",
        "review_notes": ["百科页面存在，仅证明人物/条目可追溯；风格描述、作品归属和模型标签仍需人工逐句核验。"],
    }


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    audit = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_policy": [
        "https://zh.wikipedia.org/", "https://commons.wikimedia.org/"
    ], "entries": []}
    entries = knowledge()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(audit_entry, item): index for index, item in enumerate(entries, start=1)}
        for completed, future in enumerate(as_completed(futures), start=1):
            entry = future.result()
            print(f"[{completed}/{len(entries)}] {entry['name']}")
            audit["entries"].append(entry)
            AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {AUDIT_PATH} and downloaded assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()
