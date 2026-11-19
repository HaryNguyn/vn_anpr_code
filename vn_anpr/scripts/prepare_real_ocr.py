"""Convert VietOCR labels to project CSVs, deduplicated and split by plate."""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from anpr.grammar import normalize
from anpr.rectify import rectify_plate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/real_ocr"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("Output exists; choose another --out to preserve data.")
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    charset = set(cfg["ocr"]["charset"])
    index = defaultdict(list)
    for p in sorted(args.source.rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            index[p.name].append(p)
    by_hash = {}
    rejected = []
    missing = 0
    duplicates = 0
    conflicts = set()
    for split in ("train", "val"):
        label_file = args.source / "ocr_labels" / f"{split}.txt"
        for n, line in enumerate(label_file.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) != 2:
                rejected.append({"line": n, "file": str(label_file), "reason": "invalid row"})
                continue
            name = parts[0].replace("\\", "/").split("/")[-1]
            label = normalize(parts[1].strip())
            if not label or not set(label) <= charset:
                rejected.append({"name": name, "reason": "unsupported label", "label": label})
                continue
            candidates = index.get(name, [])
            if not candidates:
                missing += 1
                continue
            for p in candidates:
                raw = p.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest in by_hash:
                    duplicates += 1
                    if by_hash[digest]["label"] != label:
                        conflicts.add(digest)
                    continue
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    rejected.append({"name": name, "reason": "image decode failed"})
                    continue
                h, w = img.shape[:2]
                layout = "one_line" if w / h >= cfg["rectify"]["one_line_min_ratio"] else "two_line"
                processed = rectify_plate(img, (0, 0, w - 1, h - 1), None,
                                          cfg["rectify"], 0.0, force_layout=layout)["ocr_input"]
                assert processed.shape == (cfg["rectify"]["ocr_height"], cfg["rectify"]["ocr_width"])
                by_hash[digest] = {"source": str(p), "label": label, "layout": layout,
                                   "width": w, "height": h, "sha256": digest}
    for digest in conflicts:
        del by_hash[digest]
    groups = defaultdict(list)
    for record in by_hash.values():
        groups[record["label"]].append(record)
    labels = sorted(groups)
    if len(labels) < 10:
        raise RuntimeError("Not enough unique labels for train/val/test.")
    random.Random(args.seed).shuffle(labels)
    nv = max(1, round(len(labels) * 0.1))
    split_labels = {"val": labels[:nv], "test": labels[nv:2 * nv], "train": labels[2 * nv:]}
    args.out.mkdir(parents=True)
    (args.out / "images").mkdir()
    manifest = []
    summary = {}
    for split, chosen in split_labels.items():
        records = [r for label in chosen for r in groups[label]]
        with (args.out / f"ocr_{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "layout"])
            for record in records:
                src = Path(record["source"])
                relative = f"images/{record['sha256']}{src.suffix.lower()}"
                (args.out / relative).write_bytes(src.read_bytes())
                writer.writerow([relative, record["label"], record["layout"]])
                manifest.append({**record, "path": relative, "split": split})
        summary[split] = {"images": len(records), "unique_plates": len(chosen),
                          "two_line": sum(r["layout"] == "two_line" for r in records)}
    report = {"source": str(args.source), "seed": args.seed, "splits": summary,
              "missing_label_rows": missing, "duplicate_records_removed": duplicates,
              "conflicting_image_labels_removed": len(conflicts), "rejected": rejected,
              "split_policy": "Disjoint normalized plate strings and exact file hashes.",
              "layout_policy": "Aspect-ratio heuristic; requires manual review for tightly cropped plates."}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
