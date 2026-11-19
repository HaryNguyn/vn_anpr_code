"""Phát hiện biển số. Hai backend:
- onnx: mô hình YOLO (v8/v11 hoặc tương thích) đã export ONNX, chạy bằng onnxruntime -> không phụ thuộc thư viện huấn luyện.
- ultralytics: chạy trực tiếp file .pt (lưu ý giấy phép AGPL-3.0).
Hỗ trợ mô hình pose 4 điểm góc biển (num_keypoints=4) để nắn phối cảnh chính xác ở góc lệch lớn.
"""
from __future__ import annotations
from typing import List
import numpy as np
import cv2
from .types import Detection


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thres]
    return keep


def letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


class OnnxYoloDetector:
    def __init__(self, cfg: dict):
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(cfg["weights"], providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.size = int(cfg.get("input_size", 640))
        self.nc = int(cfg.get("num_classes", 1))
        self.nk = int(cfg.get("num_keypoints", 0))
        self.conf = float(cfg.get("conf_thres", 0.35))
        self.iou = float(cfg.get("iou_thres", 0.5))

    def __call__(self, img: np.ndarray) -> List[Detection]:
        lb, r, left, top = letterbox(img, self.size)
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0]
        out = out.reshape(out.shape[-2], out.shape[-1])   # (C, N) hoặc (N, C), batch = 1
        expected = 4 + self.nc + 3 * self.nk
        if out.shape[0] == expected and out.shape[1] != expected:
            out = out.T                                  # (N, C)
        if out.shape[1] != expected:
            raise ValueError(f"Output ONNX shape {out.shape} không khớp 4+nc+3*nk={expected}. "
                             "Kiểm tra num_classes/num_keypoints hoặc mô hình đã có NMS tích hợp.")
        cls_scores = out[:, 4:4 + self.nc]
        cls_id = cls_scores.argmax(1)
        score = cls_scores.max(1)
        m = score >= self.conf
        out, cls_id, score = out[m], cls_id[m], score[m]
        if len(out) == 0:
            return []
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        keep = nms(boxes, score, self.iou)
        H, W = img.shape[:2]
        dets = []
        for i in keep:
            b = boxes[i].copy()
            b[[0, 2]] = ((b[[0, 2]] - left) / r).clip(0, W - 1)
            b[[1, 3]] = ((b[[1, 3]] - top) / r).clip(0, H - 1)
            kps = None
            if self.nk >= 4:
                k = out[i, 4 + self.nc:].reshape(self.nk, 3)[:4, :2].copy()
                k[:, 0] = (k[:, 0] - left) / r
                k[:, 1] = (k[:, 1] - top) / r
                kps = k
            dets.append(Detection(tuple(map(float, b)), float(score[i]), int(cls_id[i]), kps))
        return dets


class UltralyticsDetector:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO          # AGPL-3.0 – xem README mục Bản quyền
        self.model = YOLO(cfg["weights"])
        self.conf = float(cfg.get("conf_thres", 0.35))
        self.iou = float(cfg.get("iou_thres", 0.5))
        self.size = int(cfg.get("input_size", 640))

    def __call__(self, img: np.ndarray) -> List[Detection]:
        res = self.model.predict(img, conf=self.conf, iou=self.iou, imgsz=self.size, verbose=False)[0]
        dets = []
        kps_all = None
        if getattr(res, "keypoints", None) is not None and res.keypoints is not None:
            kps_all = res.keypoints.xy.cpu().numpy()
        for j, (b, s, c) in enumerate(zip(res.boxes.xyxy.cpu().numpy(),
                                          res.boxes.conf.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy())):
            k = kps_all[j][:4] if kps_all is not None and kps_all.shape[1] >= 4 else None
            dets.append(Detection(tuple(map(float, b)), float(s), int(c), k))
        return dets


def build_detector(cfg: dict):
    backend = cfg.get("backend", "onnx")
    if backend == "onnx":
        return OnnxYoloDetector(cfg)
    if backend == "ultralytics":
        return UltralyticsDetector(cfg)
    raise ValueError(f"Backend phát hiện không hỗ trợ: {backend}")
