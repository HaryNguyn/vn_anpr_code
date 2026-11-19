"""Pipeline đầy đủ: phát hiện -> nắn chỉnh -> OCR -> văn phạm -> kiểm soát chất lượng -> (video) theo vết + bỏ phiếu."""
from __future__ import annotations
import os
import re
from collections import defaultdict
from typing import Iterable, List, Optional
import cv2
import numpy as np
import yaml

from .types import PlateRead
from .rectify import rectify_plate
from .grammar import PlateGrammar
from .tracker import IoUTracker, vote

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ANPRPipeline:
    def __init__(self, cfg: dict, detector=None, recognizer=None):
        self.cfg = cfg
        if detector is None:
            from .detector import build_detector
            detector = build_detector(cfg["detector"])
        if recognizer is None:
            from .ocr import build_recognizer
            recognizer = build_recognizer(cfg["ocr"])
        self.det, self.ocr = detector, recognizer
        self.grammar = PlateGrammar(cfg.get("grammar", {}))
        self.q = cfg.get("quality", {})
        self.d = cfg.get("decision", {})
        self.crop_dir = None          # đặt thư mục để lưu crop biển phục vụ gán nhãn/huấn luyện lại
        self._crop_n = 0

    # ------------------------------------------------------------------ 1 khung hình / 1 ảnh
    def read_frame(self, img: np.ndarray, frame_idx: int = -1, source: str = "") -> List[PlateRead]:
        reads = []
        pad = float(self.cfg["detector"].get("pad_ratio", 0.1))
        for det in self.det(img):
            if self.crop_dir:
                self._save_crop(img, det.box, source, frame_idx)
            rect = rectify_plate(img, det.box, det.keypoints, self.cfg["rectify"], pad)
            ocr = self.ocr(rect["ocr_input"])
            text, pattern, conf = self.grammar.correct(ocr)
            # Thử bố cục còn lại nếu không khớp văn phạm (biển ô tô 2 hàng ngắn / biển bị cắt lệch)
            if not pattern:
                alt = "one_line" if rect["layout"] == "two_line" else "two_line"
                rect2 = rectify_plate(img, det.box, det.keypoints, self.cfg["rectify"], pad, force_layout=alt)
                ocr2 = self.ocr(rect2["ocr_input"])
                t2, p2, c2 = self.grammar.correct(ocr2)
                if p2:
                    rect, ocr, text, pattern, conf = rect2, ocr2, t2, p2, c2
            r = self._make_read(det, rect, ocr.text, text, pattern, conf, frame_idx, source)
            r.evidence = rect["warp"]
            reads.append(r)
        return reads

    def _save_crop(self, img, box, source, frame_idx):
        x1, y1, x2, y2 = map(int, box)
        px, py = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
        crop = img[max(0, y1 - py):y2 + py, max(0, x1 - px):x2 + px]
        if crop.size:
            os.makedirs(self.crop_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(source))[0] or "frame"
            base = re.sub(r'[<>:"/\\|?*]', "_", base)          # ký tự cấm trong tên file Windows
            cv2.imwrite(os.path.join(self.crop_dir, f"{base}_{frame_idx}_{self._crop_n:06d}.jpg"), crop)
            self._crop_n += 1

    def _make_read(self, det, rect, raw, text, pattern, conf, frame_idx, source) -> PlateRead:
        min_h = float(self.q.get("min_plate_height_px", 20))
        min_s = float(self.q.get("min_sharpness", 30.0))
        max_skew = float(self.q.get("max_skew_deg", 30.0))
        ph, sh, sk = rect["plate_height_px"], rect["sharpness"], rect["skew_deg"]
        quality = min(1.0, ph / (2 * min_h)) * min(1.0, sh / (3 * min_s)) * (1.0 if sk <= max_skew else 0.5)
        if ph < min_h or sh < min_s:
            status = "LOW_QUALITY"
        elif pattern and conf >= float(self.d.get("accept_conf", 0.9)):
            status = "ACCEPT"
        else:
            status = "UNREAD"
        return PlateRead(det.box, det.score, rect["layout"], raw, text, float(conf), bool(pattern),
                         pattern, float(quality), ph, sh, sk, status, frame_idx, -1, source)

    # ------------------------------------------------------------------ nhiều ảnh
    def process_images(self, paths: Iterable[str], group_by_folder: bool = False) -> dict:
        per_image, groups = [], defaultdict(list)
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                per_image.append({"source": p, "status": "READ_ERROR"}); continue
            reads = self.read_frame(img, -1, p)
            if not reads:
                per_image.append({"source": p, "status": "NO_PLATE", "text": ""})
            for r in reads:
                per_image.append(_read_to_dict(r))
                if group_by_folder:
                    groups[os.path.dirname(p)].append(r)
        out = {"images": per_image}
        if group_by_folder:
            out["groups"] = [self._decide(k, v, min_reads=1) for k, v in groups.items()]
        return out

    # ------------------------------------------------------------------ video / luồng camera
    def new_session(self, fps: float = 25.0) -> "VideoSession":
        return VideoSession(self, fps)

    def process_video(self, path: str, out_video: Optional[str] = None, on_vehicle=None) -> dict:
        """Xử lý file video hoặc RTSP đến hết luồng. on_vehicle(result: dict, evidence: ndarray|None)
        được gọi ngay khi một xe kết thúc (rời khung hình) – dùng để đẩy sự kiện sang phần mềm bãi đỗ."""
        stride = max(1, int(self.cfg.get("video", {}).get("stride", 1)))
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Không mở được video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sess = self.new_session(fps)
        writer, results, idx = None, [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                done, reads = sess.feed(frame, idx, path)
                for res, ev in done:
                    results.append(res)
                    if on_vehicle:
                        on_vehicle(res, ev)
                if out_video:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
                    writer.write(_draw(frame, reads))
            idx += 1
        for res, ev in sess.flush():
            results.append(res)
            if on_vehicle:
                on_vehicle(res, ev)
        cap.release()
        if writer is not None:
            writer.release()
        return {"video": path, "frames": idx, "fps": fps, "vehicles": results}

    def decide_reads(self, reads: List[PlateRead], key="request") -> dict:
        """Quyết định cuối cho một nhóm ảnh của CÙNG một xe (API chụp theo trigger)."""
        return self._decide(key, reads, min_reads=1)

    def _decide(self, key, reads: List[PlateRead], min_reads: Optional[int] = None,
                fps: float = 0.0, first: int = -1, last: int = -1) -> dict:
        usable = [r for r in reads if r.status != "LOW_QUALITY"] or reads
        v = vote(usable)
        need = int(self.d.get("min_reads_video", 2)) if min_reads is None else min_reads
        # Chốt khi: hợp lệ văn phạm + các lần đọc ủng hộ đủ tin cậy + đủ đồng thuận + đủ số lần đọc
        accept = (v["grammar_ok"] and v.get("mean_conf", 0.0) >= float(self.d.get("accept_conf", 0.9))
                  and v.get("agree", 0.0) >= float(self.d.get("min_agree", 0.7))
                  and v["support"] >= min(need, len(usable)))
        best = max(usable, key=lambda r: r.quality * r.conf) if usable else None
        return {"key": key, "text": v["text"] if accept else "", "candidate": v["text"],
                "status": "ACCEPT" if accept else "UNREAD", "conf": round(v["conf"], 4),
                "agree": round(v.get("agree", 0.0), 4), "support": v["support"], "n_reads": v["n_reads"],
                "layout": best.layout if best else "", "pattern": best.pattern if best else "",
                "first_time_s": round(first / fps, 2) if fps and first >= 0 else None,
                "last_time_s": round(last / fps, 2) if fps and last >= 0 else None}


def _read_to_dict(r: PlateRead) -> dict:
    return {"source": r.source, "frame": r.frame_idx, "status": r.status,
            "text": r.text if r.status == "ACCEPT" else "", "candidate": r.text, "raw": r.text_raw,
            "conf": round(r.conf, 4), "pattern": r.pattern, "layout": r.layout,
            "plate_h_px": round(r.plate_height_px, 1), "sharpness": round(r.sharpness, 1),
            "skew_deg": round(r.skew_deg, 1), "box": [round(x, 1) for x in r.box]}


def _draw(frame: np.ndarray, reads: List[PlateRead]) -> np.ndarray:
    vis = frame.copy()
    for r in reads:
        x1, y1, x2, y2 = map(int, r.box)
        color = (0, 200, 0) if r.status == "ACCEPT" else (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{r.text or '?'} {r.conf:.2f} T{r.track_id}"
        cv2.putText(vis, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def list_images(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _, files in os.walk(path):
        out += [os.path.join(root, f) for f in sorted(files) if os.path.splitext(f)[1].lower() in IMG_EXT]
    return sorted(out)


class VideoSession:
    """Phiên theo vết cho MỘT camera/làn. Gọi feed() cho từng khung hình; nhận về các xe đã chốt."""
    def __init__(self, pipe: "ANPRPipeline", fps: float = 25.0):
        vc = pipe.cfg.get("video", {})
        self.pipe, self.fps = pipe, fps
        self.tracker = IoUTracker(float(vc.get("iou_match", 0.3)), int(vc.get("max_age", 15)),
                                  int(vc.get("max_reads_per_track", 15)))

    def feed(self, frame: np.ndarray, frame_idx: int, source: str = ""):
        reads = self.pipe.read_frame(frame, frame_idx, source)
        done = [self._finish(t) for t in self.tracker.update(frame_idx, reads)]
        return done, reads

    def flush(self):
        return [self._finish(t) for t in self.tracker.flush()]

    def _finish(self, t):
        res = self.pipe._decide(t.tid, t.reads, fps=self.fps, first=t.first_frame, last=t.last_frame)
        best = max(t.reads, key=lambda r: r.quality * r.conf) if t.reads else None
        return res, (best.evidence if best is not None else None)
