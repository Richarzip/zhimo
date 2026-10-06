"""Compare raw/basic/adaptive pixels and optional recognition, without training.

The input is a pixel reference for synthetic corruption, not an author label.
Recognition output measures prediction changes only; it is not an accuracy test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zhimo.vision.denoise import denoise_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "image_test" / "test.png")
    parser.add_argument("--out", type=Path, required=True, help="Directory for JSON and comparison PNG")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recognize", action="store_true", help="Also infer using deployed weights; never trains or calls an API")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as image:
        reference = ImageOps.exif_transpose(image).convert("RGB")
    # Keep this quick comparison bounded; production denoising preserves size.
    reference.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    clean = np.asarray(reference)
    rng = np.random.default_rng(args.seed)
    salt = clean.copy()
    sample = rng.random(clean.shape[:2])
    salt[sample < 0.008] = 0
    salt[(sample >= 0.008) & (sample < 0.016)] = 255
    variants = {
        "clean": clean,
        "gaussian": np.clip(clean.astype(float) + rng.normal(0, 12, clean.shape), 0, 255).astype(np.uint8),
        "impulse": salt,
        "shadow": np.rint(clean * np.linspace(0.7, 1, clean.shape[1])[None, :, None]).astype(np.uint8),
        "faded": np.rint(0.25 * clean + 175).astype(np.uint8),
    }
    recognizer = None
    if args.recognize:
        from zhimo.application import get_recognizer
        recognizer = get_recognizer()
    records = []
    board = Image.new("RGB", (4 * 280, len(variants) * 300), "white")
    draw = ImageDraw.Draw(board)
    for row, (name, array) in enumerate(variants.items()):
        raw = Image.fromarray(array)
        panels = [("reference", reference)]
        for mode in ("raw", "basic", "adaptive"):
            started = time.perf_counter()
            output, diagnostics = (raw, {}) if mode == "raw" else denoise_image(raw, mode=mode)
            milliseconds = (time.perf_counter() - started) * 1000
            mse = float(np.mean((np.asarray(output, dtype=float) - clean.astype(float)) ** 2))
            record = {"case": name, "mode": mode, "mse": round(mse, 4),
                      "psnr_db": round(float(10 * np.log10(255 ** 2 / mse)), 3) if mse else None,
                      "processing_ms": round(milliseconds, 3), "diagnostics": diagnostics}
            if recognizer:
                result = recognizer.recognize(output)
                record["recognition"] = {key: result.get(key) for key in ("calligrapher", "confidence", "ensemble")}
            records.append(record)
            panels.append((mode, output))
        for col, (label, panel) in enumerate(panels):
            preview = panel.copy()
            preview.thumbnail((272, 266))
            board.paste(preview, (col * 280 + 4, row * 300 + 25))
            draw.text((col * 280 + 4, row * 300 + 5), f"{name}: {label}", fill="black")
    report = {"reference": str(args.image.resolve()), "seed": args.seed,
              "note": "Synthetic pixel comparison and optional inference smoke; no author ground truth or accuracy claim.",
              "records": records}
    (args.out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    board.save(args.out / "comparison.png")
    print(json.dumps({"cases": len(variants), "variants": len(records), "output": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
