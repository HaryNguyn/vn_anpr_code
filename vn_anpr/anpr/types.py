from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    box: Tuple[float, float, float, float]          # x1, y1, x2, y2 (pixel ảnh gốc)
    score: float
    cls: int = 0
    keypoints: Optional[np.ndarray] = None          # (4, 2) nếu mô hình trả 4 góc


@dataclass
class OcrResult:
    text: str
    conf: float                                     # độ tin cậy mức biển = tích (hoặc min) xác suất ký tự
    char_conf: List[float] = field(default_factory=list)
    char_topk: List[List[Tuple[str, float]]] = field(default_factory=list)


@dataclass
class PlateRead:
    box: Tuple[float, float, float, float]
    det_score: float
    layout: str                                     # one_line | two_line
    text_raw: str
    text: str                                       # sau hậu xử lý văn phạm
    conf: float
    grammar_ok: bool
    pattern: str
    quality: float                                  # trọng số chất lượng 0..1
    plate_height_px: float
    sharpness: float
    skew_deg: float
    status: str                                     # ACCEPT | UNREAD | LOW_QUALITY
    frame_idx: int = -1
    track_id: int = -1
    source: str = ""
    evidence: Optional[np.ndarray] = field(default=None, repr=False)   # ảnh crop biển (bằng chứng)
