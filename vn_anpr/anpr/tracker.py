"""Theo vết biển số giữa các khung hình (IoU + khoảng cách tâm) và bỏ phiếu đa khung hình.
Đủ cho làn có barie, cổng bãi đỗ, kiểm soát ra vào. Với cao tốc/MLFF tốc độ cao nên thay bằng ByteTrack/OC-SORT
có lọc Kalman [Nhận định].
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
from .types import PlateRead


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    tid: int
    box: tuple
    last_frame: int
    first_frame: int
    reads: List[PlateRead] = field(default_factory=list)


class IoUTracker:
    def __init__(self, iou_match: float = 0.3, max_age: int = 15, max_reads: int = 15):
        self.iou_match, self.max_age, self.max_reads = iou_match, max_age, max_reads
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, frame_idx: int, reads: List[PlateRead]) -> List[Track]:
        """Gán các lần đọc vào track; trả các track đã kết thúc (mất dấu quá max_age)."""
        unmatched = list(range(len(reads)))
        pairs = []
        for tid, t in self.tracks.items():
            for j in unmatched:
                s = iou(t.box, reads[j].box)
                if s < self.iou_match:   # dự phòng: xe dịch chuyển nhanh -> xét khoảng cách tâm
                    s = self._center_score(t.box, reads[j].box)
                if s >= self.iou_match:
                    pairs.append((s, tid, j))
        used_t, used_r = set(), set()
        for s, tid, j in sorted(pairs, reverse=True):
            if tid in used_t or j in used_r:
                continue
            used_t.add(tid); used_r.add(j)
            self._add(self.tracks[tid], reads[j], frame_idx)
        for j in unmatched:
            if j not in used_r:
                t = Track(self.next_id, reads[j].box, frame_idx, frame_idx)
                self.tracks[self.next_id] = t
                self._add(t, reads[j], frame_idx)
                self.next_id += 1
        finished = [t for t in self.tracks.values() if frame_idx - t.last_frame > self.max_age]
        for t in finished:
            del self.tracks[t.tid]
        return finished

    def flush(self) -> List[Track]:
        out = list(self.tracks.values()); self.tracks.clear(); return out

    def _add(self, t: Track, r: PlateRead, frame_idx: int):
        r.track_id = t.tid
        t.box, t.last_frame = r.box, frame_idx
        t.reads.append(r)
        if len(t.reads) > self.max_reads:            # giữ các lần đọc chất lượng cao nhất
            t.reads.sort(key=lambda x: x.quality * x.conf, reverse=True)
            t.reads = t.reads[: self.max_reads]

    @staticmethod
    def _center_score(a, b) -> float:
        ca = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
        cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        diag = np.hypot(a[2] - a[0], a[3] - a[1]) + 1e-6
        d = np.linalg.norm(ca - cb) / diag
        return max(0.0, 1.0 - d) * 0.5   # tối đa 0.5 để ưu tiên ghép bằng IoU


def vote(reads: List[PlateRead]) -> dict:
    """Bỏ phiếu có trọng số = conf * chất lượng ảnh; ưu tiên chuỗi hợp lệ văn phạm.
    Bổ sung bỏ phiếu theo từng vị trí ký tự trên các chuỗi cùng độ dài trội nhất."""
    valid = [r for r in reads if r.text]
    if not valid:
        return {"text": "", "conf": 0.0, "support": 0, "n_reads": len(reads), "grammar_ok": False}
    pool = [r for r in valid if r.grammar_ok] or valid
    w = lambda r: max(r.conf, 1e-3) * max(r.quality, 1e-3)
    scores = defaultdict(float)
    for r in pool:
        scores[r.text] += w(r)
    total = sum(scores.values())
    best_text, best_score = max(scores.items(), key=lambda kv: kv[1])
    # bỏ phiếu theo ký tự
    lens = defaultdict(float)
    for r in pool:
        lens[len(r.text)] += w(r)
    L = max(lens, key=lens.get)
    same = [r for r in pool if len(r.text) == L]
    char_text = ""
    for i in range(L):
        cs = defaultdict(float)
        for r in same:
            cs[r.text[i]] += w(r)
        char_text += max(cs, key=cs.get)
    if char_text != best_text and any(r.text == char_text for r in pool):
        best_text, best_score = char_text, scores[char_text]
    sup = [r for r in pool if r.text == best_text]
    grammar_ok = any(r.grammar_ok for r in sup)
    # độ tin cậy cuối = tỷ lệ đồng thuận x độ tin cậy trung bình của các lần đọc ủng hộ
    agree = best_score / total if total else 0.0
    mean_conf = float(np.mean([r.conf for r in sup])) if sup else 0.0
    return {"text": best_text, "conf": agree * mean_conf, "agree": agree, "mean_conf": mean_conf,
            "support": len(sup), "n_reads": len(reads), "grammar_ok": grammar_ok}
