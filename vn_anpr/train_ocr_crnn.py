#!/usr/bin/env python3
"""Huấn luyện mô hình OCR biển số CRNN + CTC (PyTorch) và export ONNX cho anpr/ocr.py.

Dữ liệu: CSV có cột  path,label[,layout]
  - path : ảnh biển đã cắt (crop) từ detector (run_anpr.py --save-crops), KHÔNG cần nắn trước
  - label: chuỗi biển, ví dụ "30A-123.45" hoặc "29-B1 123.45" (tự chuẩn hóa còn chữ và số)
  - layout (tùy chọn): one_line | two_line – nếu không có sẽ tự xác định như lúc suy luận
Tiền xử lý dùng CHUNG hàm rectify_plate/to_ocr_input với suy luận để tránh lệch miền.

Ví dụ:
  python train_ocr_crnn.py --train data/ocr_train.csv --val data/ocr_val.csv --epochs 80 --out models/
YÊU CẦU CHIA TẬP: tách train/val/test theo TRẠM/CAMERA và NGÀY, không chia ngẫu nhiên theo ảnh
(tránh rò rỉ ảnh gần trùng của cùng một xe giữa các tập).
"""
import argparse, csv, os, random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

from anpr.rectify import rectify_plate
from anpr.grammar import normalize
from anpr.ocr import ctc_greedy_decode


class PlateDataset(Dataset):
    def __init__(self, csv_path, charset, rect_cfg, augment=False):
        self.items, self.charset, self.rect_cfg, self.augment = [], charset, rect_cfg, augment
        self.c2i = {c: i + 1 for i, c in enumerate(charset)}
        root = os.path.dirname(os.path.abspath(csv_path))
        with open(csv_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                lab = normalize(r["label"])
                if not lab or any(c not in self.c2i for c in lab):
                    print("Bỏ qua nhãn không hợp lệ:", r["label"]); continue
                p = r["path"] if os.path.isabs(r["path"]) else os.path.join(root, r["path"])
                self.items.append((p, lab, r.get("layout") or None))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab, layout = self.items[i]
        img = cv2.imread(p)
        if img is None:
            raise IOError(p)
        if self.augment:
            img = aug_color(img)
        h, w = img.shape[:2]
        x = rectify_plate(img, (0, 0, w - 1, h - 1), None, self.rect_cfg, 0.0, force_layout=layout)["ocr_input"]
        if self.augment:
            x = aug_geom(x)
        x = (x.astype(np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(x)[None], torch.tensor([self.c2i[c] for c in lab]), lab


def aug_color(img):
    if random.random() < 0.8:
        a, b = random.uniform(0.7, 1.3), random.uniform(-30, 30)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        k = random.choice([3, 5]); img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.3:
        q = random.randint(30, 80)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    return img


def aug_geom(x):
    h, w = x.shape
    if random.random() < 0.5:   # sai số nắn chỉnh còn dư
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        j = lambda s: random.uniform(-s, s)
        dst = src + np.float32([[j(4), j(3)], [j(4), j(3)], [j(4), j(3)], [j(4), j(3)]])
        x = cv2.warpPerspective(x, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.3:
        x = np.clip(x + np.random.normal(0, 8, x.shape), 0, 255).astype(np.uint8)
    return x


def collate(batch):
    xs, ys, labs = zip(*batch)
    return torch.stack(xs), torch.cat(ys), torch.tensor([len(y) for y in ys]), labs


def cbr(i, o):
    return [nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    def __init__(self, nclass, in_ch=1):
        super().__init__()
        self.cnn = nn.Sequential(*cbr(in_ch, 64), nn.MaxPool2d(2, 2),          # 24x96
                                 *cbr(64, 128), nn.MaxPool2d(2, 2),            # 12x48
                                 *cbr(128, 256), *cbr(256, 256), nn.MaxPool2d((2, 1)),  # 6x48
                                 *cbr(256, 256), *cbr(256, 256), nn.MaxPool2d((2, 1)),  # 3x48
                                 *cbr(256, 256), nn.Dropout2d(0.1))
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(256, nclass)

    def forward(self, x):
        f = self.cnn(x).mean(2)              # (B, 256, T)
        f = f.permute(2, 0, 1)               # (T, B, 256)
        y, _ = self.rnn(f)
        return self.fc(y).log_softmax(-1)    # (T, B, C)


def evaluate(model, loader, charset, dev):
    model.eval(); ok = n = 0
    with torch.no_grad():
        for x, _, _, labs in loader:
            out = model(x.to(dev)).exp().cpu().numpy()
            for b, lab in enumerate(labs):
                ok += ctc_greedy_decode(out[:, b], charset).text == lab; n += 1
    return ok / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--out", default="models")
    ap.add_argument("--init", help="Checkpoint CRNN để fine-tune (cùng charset và kiến trúc)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    charset, rect = cfg["ocr"]["charset"], cfg["rectify"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(PlateDataset(a.train, charset, rect, True), a.batch, True, collate_fn=collate, num_workers=4)
    va = DataLoader(PlateDataset(a.val, charset, rect, False), a.batch, False, collate_fn=collate, num_workers=4)
    model = CRNN(len(charset) + 1).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=max(1, len(tr)))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    os.makedirs(a.out, exist_ok=True); best = -1
    for ep in range(a.epochs):
        model.train(); tot = 0
        for x, y, yl, _ in tr:
            out = model(x.to(dev))
            il = torch.full((x.size(0),), out.size(0), dtype=torch.long)
            loss = ctc(out, y.to(dev), il, yl)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); tot += loss.item()
        acc = evaluate(model, va, charset, dev)
        print(f"epoch {ep+1}/{a.epochs} loss={tot/max(1,len(tr)):.4f} val_exact={acc:.4f}")
        if acc > best:
            best = acc; torch.save(model.state_dict(), os.path.join(a.out, "plate_ocr_best.pt"))
    model.load_state_dict(torch.load(os.path.join(a.out, "plate_ocr_best.pt"), map_location=dev))
    model.eval().cpu()
    H, W = rect.get("ocr_height", 48), rect.get("ocr_width", 192)
    torch.onnx.export(model, torch.zeros(1, 1, H, W), os.path.join(a.out, "plate_ocr.onnx"),
                      input_names=["image"], output_names=["logits"], opset_version=17,
                      dynamo=False)
    print(f"Tốt nhất val_exact={best:.4f}. Đã export {a.out}/plate_ocr.onnx")


if __name__ == "__main__":
    main()
