"""Generate synthetic OCR starter data; not a real-world evaluation dataset."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def render(label, layout, rng):
    h, w = (140, 190) if layout == "two_line" else (110, 520)
    color = (240, 240, 240) if rng.random() < 0.75 else (40, 215, 240)
    image = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.rectangle(image, (3, 3), (w - 4, h - 4), (35, 35, 35), 2)
    lines = [label[:-5], label[-5:]] if layout == "two_line" else [label]
    font = int(rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]))
    thickness = int(rng.integers(2, 4))
    band = h // len(lines)
    for i, text in enumerate(lines):
        (tw, th), baseline = cv2.getTextSize(text, font, 1, thickness)
        scale = min((w - 22) / tw, (band - 20) / (th + baseline))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(image, text, ((w - tw) // 2, i * band + (band + th) // 2),
                    font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3), 0.96)
    image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.4:
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.3, 0.8))
    image = np.clip(image.astype(float) * rng.uniform(0.8, 1.05)
                    + rng.normal(0, rng.uniform(0, 5), image.shape), 0, 255).astype(np.uint8)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_ocr"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 10:
        parser.error("--count must be at least 10")
    if args.out.exists():
        parser.error("Output directory already exists; choose a new --out to preserve data.")
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    seen = set()
    counts = {}
    for split, count in [("train", args.count - args.count // 10),
                         ("val", args.count // 10)]:
        folder = args.out / split
        folder.mkdir()
        with (args.out / f"ocr_{split}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "layout"])
            for i in range(count):
                layout = "one_line" if i % 2 == 0 else "two_line"
                while True:
                    # Illustrative formats, not an authoritative registration-number generator.
                    prefix = str(rng.choice([29, 30, 43, 51, 59, 60, 61, 65]))
                    series = str(rng.choice(list("ABCDEFGHKLMNPRSTUVXYZ")))
                    if layout == "two_line":
                        series += str(rng.integers(1, 10))
                    label = f"{prefix}{series}{rng.integers(0, 100000):05d}"
                    if label not in seen:
                        seen.add(label)
                        break
                relative = f"{split}/{i:06d}.jpg"
                image = render(label, layout, rng)
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise RuntimeError(f"Cannot encode {relative}")
                (args.out / relative).write_bytes(encoded.tobytes())
                writer.writerow([relative, label, layout])
        counts[split] = count
    (args.out / "metadata.json").write_text(json.dumps({
        "synthetic": True, "seed": args.seed, "counts": counts,
        "fonts": "OpenCV Hershey; not authentic Vietnamese plate typography",
        "evaluation": "Synthetic validation only; use independent real images for acceptance.",
    }, indent=2), encoding="utf-8")
    print(f"Created {counts} in {args.out}")


if __name__ == "__main__":
    main()
