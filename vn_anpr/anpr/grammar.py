"""Hậu xử lý theo văn phạm biển số Việt Nam (Điều 37, Phụ lục 02-04 TT 79/2024/TT-BCA).
Chiến lược: (1) khớp trực tiếp; (2) tìm tổ hợp top-k ký tự có xác suất cao nhất thỏa văn phạm;
(3) sửa nhầm lẫn hình dạng theo vị trí (O<->0, B<->8...) có phạt độ tin cậy.
"""
from __future__ import annotations
import heapq
import math
import re
from typing import List, Tuple
from .types import OcrResult

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "T": "1", "Z": "2",
            "S": "5", "G": "6", "B": "8", "A": "4"}
TO_ALPHA = {"0": "D", "1": "T", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}


def normalize(text: str) -> str:
    text = text.upper().replace("Ð", "Đ")
    return "".join(ch for ch in text if ch.isalnum())


class PlateGrammar:
    def __init__(self, cfg: dict):
        self.patterns = [(p["name"], re.compile(p["regex"])) for p in cfg.get("patterns", [])]
        self.provinces = set(str(x).zfill(2) for x in cfg.get("valid_province_codes", []) or [])

    def match(self, text: str) -> str:
        for name, rx in self.patterns:
            if rx.match(text) and (not self.provinces or text[:2] in self.provinces):
                return name
        return ""

    def correct(self, r: OcrResult, max_expand: int = 300) -> Tuple[str, str, float]:
        text = normalize(r.text)
        name = self.match(text)
        if name:
            return text, name, r.conf
        # (2) tìm theo top-k (chỉ khi số ký tự giữ nguyên sau chuẩn hóa)
        if r.char_topk and len(r.char_topk) == len(r.text):
            res = self._search_topk(r.char_topk, max_expand)
            if res:
                return res
        # (3) sửa nhầm lẫn theo vị trí
        fixed = self._positional_fix(text)
        for cand in fixed:
            name = self.match(cand)
            if name:
                return cand, name, r.conf * 0.8
        return text, "", r.conf

    def _search_topk(self, topk: List[List[Tuple[str, float]]], max_expand: int):
        cands = [[(c, p) for c, p in alts if p >= 0.02] or alts[:1] for alts in topk]
        start = tuple([0] * len(cands))
        score = lambda idx: -sum(math.log(max(cands[i][j][1], 1e-9)) for i, j in enumerate(idx))
        heap, seen, n = [(score(start), start)], {start}, 0
        while heap and n < max_expand:
            s, idx = heapq.heappop(heap); n += 1
            text = normalize("".join(cands[i][j][0] for i, j in enumerate(idx)))
            name = self.match(text)
            if name:
                conf = min(cands[i][j][1] for i, j in enumerate(idx))
                return text, name, conf
            for i in range(len(idx)):
                if idx[i] + 1 < len(cands[i]):
                    nxt = idx[:i] + (idx[i] + 1,) + idx[i + 1:]
                    if nxt not in seen:
                        seen.add(nxt); heapq.heappush(heap, (score(nxt), nxt))
        return None

    @staticmethod
    def _positional_fix(text: str) -> List[str]:
        """Sinh ứng viên: 2 ký tự đầu là số (mã địa phương), ký tự thứ 3 là chữ (seri),
        4-5 ký tự cuối là số (thứ tự đăng ký)."""
        out = []
        if len(text) < 6:
            return out
        for tail in (5, 4):
            if len(text) < 3 + tail:
                continue
            t = list(text)
            for i in (0, 1):
                t[i] = TO_DIGIT.get(t[i], t[i])
            t[2] = TO_ALPHA.get(t[2], t[2])
            for i in range(len(t) - tail, len(t)):
                t[i] = TO_DIGIT.get(t[i], t[i])
            out.append("".join(t))
        return out
