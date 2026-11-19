"""Nhận dạng ký tự biển số không phân đoạn (segmentation-free) bằng CTC.
Mô hình: CRNN/LPRNet/SVTR... bất kỳ, export ONNX, đầu ra (T, C) hoặc (T,1,C) hoặc (1,T,C), C = len(charset)+1, blank = 0.
Trả về độ tin cậy từng ký tự và top-k thay thế để hậu xử lý văn phạm chọn phương án hợp lệ.
"""
from __future__ import annotations
import numpy as np
from .types import OcrResult


def _to_probs(logits: np.ndarray) -> np.ndarray:
    if logits.max() <= 0.0:                         # log-softmax
        return np.exp(logits)
    s = logits.sum(-1)
    if np.allclose(s, 1.0, atol=1e-3) and logits.min() >= 0:
        return logits
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def ctc_greedy_decode(probs: np.ndarray, charset: str, topk: int = 3) -> OcrResult:
    """probs: (T, C) với C = len(charset) + 1, chỉ số 0 = blank."""
    best = probs.argmax(1)
    chars, confs, alts = [], [], []
    prev = 0
    for t, k in enumerate(best):
        p = float(probs[t, k])
        if k != 0 and k != prev:
            chars.append(charset[k - 1]); confs.append(p)
            alts.append(_topk(probs[t], charset, topk))
        elif k != 0 and k == prev and p > confs[-1]:
            confs[-1] = p
            alts[-1] = _topk(probs[t], charset, topk)
        prev = k
    text = "".join(chars)
    conf = float(min(confs)) if confs else 0.0
    return OcrResult(text, conf, confs, alts)


def _topk(row: np.ndarray, charset: str, k: int):
    idx = row[1:].argsort()[::-1][:k]
    return [(charset[i], float(row[i + 1])) for i in idx]


class OnnxCtcRecognizer:
    def __init__(self, cfg: dict):
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(cfg["weights"], providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.charset = cfg["charset"]
        self.ch = int(cfg.get("input_channels", 1))
        self.topk = int(cfg.get("topk", 3))

    def preprocess(self, gray: np.ndarray) -> np.ndarray:
        x = gray.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = x[None, None] if self.ch == 1 else np.repeat(x[None, None], 3, 1)
        return x

    def __call__(self, gray: np.ndarray) -> OcrResult:
        out = self.sess.run(None, {self.inp: self.preprocess(gray)})[0]
        C = len(self.charset) + 1
        out = np.squeeze(out)
        if out.ndim != 2:
            raise ValueError(f"Đầu ra OCR không hợp lệ: {out.shape}")
        if out.shape[1] != C and out.shape[0] == C:
            out = out.T
        if out.shape[1] != C:
            raise ValueError(f"Số lớp đầu ra {out.shape} khác len(charset)+1={C}; kiểm tra charset.")
        return ctc_greedy_decode(_to_probs(out), self.charset, self.topk)


def build_recognizer(cfg: dict):
    if cfg.get("backend", "onnx_ctc") == "onnx_ctc":
        return OnnxCtcRecognizer(cfg)
    raise ValueError(f"Backend OCR không hỗ trợ: {cfg.get('backend')}")
