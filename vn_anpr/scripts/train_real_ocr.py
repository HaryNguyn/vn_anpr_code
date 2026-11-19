"""Train in a staging directory, validate ONNX, report test metrics and deploy."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import onnx
import torch
import yaml

from anpr.ocr import OnnxCtcRecognizer
from anpr.rectify import rectify_plate
from train_ocr_crnn import CRNN


def evaluate(path, csv_path, cfg):
    ocr = OnnxCtcRecognizer({**cfg["ocr"], "weights": str(path)})
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    correct = 0
    predictions = []
    for row in rows:
        img = cv2.imdecode(np.frombuffer((csv_path.parent / row["path"]).read_bytes(), np.uint8), 1)
        h, w = img.shape[:2]
        gray = rectify_plate(img, (0, 0, w - 1, h - 1), None, cfg["rectify"], 0.0,
                             force_layout=row["layout"])["ocr_input"]
        result = ocr(gray)
        correct += result.text == row["label"]
        predictions.append({"path": row["path"], "label": row["label"],
                            "prediction": result.text, "conf": result.conf, "layout": row["layout"]})
    return {"images": len(rows), "correct": correct, "exact_match": correct / len(rows)}, predictions


def main():
    os.chdir(ROOT)
    torch.set_num_threads(6)
    data = ROOT / "data/real_ocr"
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    # Enforce disjoint labels and file content before launching expensive training.
    groups = {}
    for split in ("train", "val", "test"):
        rows = list(csv.DictReader((data / f"ocr_{split}.csv").open(encoding="utf-8")))
        assert rows, split
        groups[split] = ({r["label"] for r in rows}, {r["path"] for r in rows})
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not groups[a][0] & groups[b][0], "Plate leakage"
        assert not groups[a][1] & groups[b][1], "Image leakage"
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = ROOT / "runtime" / f"real_ocr_{tag}"
    run.mkdir(parents=True)
    models = ROOT / "models"
    backup = models / "backups" / tag
    backup.mkdir(parents=True)
    for name in ("plate_ocr.onnx", "plate_ocr_best.pt"):
        shutil.copy2(models / name, backup / name)
    shutil.copy2(ROOT / "config.yaml", run / "config.yaml")
    shutil.copy2(data / "report.json", run / "dataset_report.json")
    print(f"Run: {run}\nBackup: {backup}", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6"}
    cmd = [sys.executable, "-u", "train_ocr_crnn.py",
           "--train", str(data / "ocr_train.csv"), "--val", str(data / "ocr_val.csv"),
           "--epochs", "80", "--batch", "32", "--lr", "0.0003",
           "--init", str(backup / "plate_ocr_best.pt"), "--out", str(run)]
    with (run / "train.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", env=env)
        for line in proc.stdout:
            log.write(line); log.flush()
            print(line, end="", flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"Training failed; existing model preserved. See {run / 'train.log'}")
    new = run / "plate_ocr.onnx"
    onnx.checker.check_model(onnx.load(str(new)))
    recognizer = OnnxCtcRecognizer({**cfg["ocr"], "weights": str(new)})
    pt = CRNN(len(cfg["ocr"]["charset"]) + 1)
    pt.load_state_dict(torch.load(run / "plate_ocr_best.pt", map_location="cpu", weights_only=True))
    pt.eval()
    x = np.random.default_rng(42).uniform(-1, 1, (1, 1, 48, 192)).astype(np.float32)
    with torch.no_grad():
        expected = pt(torch.from_numpy(x)).numpy()
    actual = recognizer.sess.run(None, {recognizer.inp: x})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    del recognizer
    old_val, _ = evaluate(backup / "plate_ocr.onnx", data / "ocr_val.csv", cfg)
    new_val, _ = evaluate(new, data / "ocr_val.csv", cfg)
    old_test, _ = evaluate(backup / "plate_ocr.onnx", data / "ocr_test.csv", cfg)
    new_test, predictions = evaluate(new, data / "ocr_test.csv", cfg)
    # Deployment decision uses validation, never the held-out test score.
    deploy = new_val["exact_match"] > old_val["exact_match"]
    report = {"backup": str(backup), "run": str(run), "old_validation": old_val,
              "new_validation": new_val, "old_test": old_test, "new_test": new_test,
              "onnx_parity": "passed", "deployed": False,
              "deployment_rule": "New validation exact match must exceed old model.",
              "scope": "OCR on cropped real plates; not end-to-end detection accuracy."}
    (run / "test_predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    (run / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if deploy:
        for name in ("plate_ocr_best.pt", "plate_ocr.onnx"):
            temp = models / (name + ".pending")
            shutil.copy2(run / name, temp)
            os.replace(temp, models / name)
        report["deployed"] = True
    (run / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (models / "plate_ocr.training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
