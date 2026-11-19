"""Nắn chỉnh phối cảnh biển số (homography) và chuẩn hóa ảnh đầu vào OCR.
Mục tiêu: triệt tiêu méo phối cảnh khi camera lệch phương vuông góc tới 30 độ.
Thứ tự ưu tiên: 4 góc từ mô hình pose  >  tứ giác tìm bằng đường viền  >  minAreaRect  >  hộp bao trục.
Hàm to_ocr_input() dùng CHUNG cho huấn luyện và suy luận để tránh lệch tiền xử lý.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
import cv2


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sắp 4 điểm theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def find_quad(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Tìm tứ giác viền biển trong vùng cắt (tọa độ cục bộ)."""
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    min_area = 0.25 * h * w
    candidates = []
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=1)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for binimg in (edges, th, 255 - th):
        cnts, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.02, 0.03, 0.045, 0.06):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4 and cv2.isContourConvex(ap):
                    candidates.append((cv2.contourArea(ap), ap.reshape(4, 2)))
                    break
            else:
                rect = cv2.minAreaRect(hull)
                candidates.append((0.9 * rect[1][0] * rect[1][1], cv2.boxPoints(rect)))
    # loại tứ giác là nền/khung crop: >= 2 đỉnh nằm sát mép vùng cắt hoặc chiếm > 92% diện tích
    def on_border(q):
        q = np.asarray(q)
        return int(((q[:, 0] <= 1) | (q[:, 0] >= w - 2) | (q[:, 1] <= 1) | (q[:, 1] >= h - 2)).sum())
    candidates = [c for c in candidates if on_border(c[1]) < 2 and c[0] <= 0.92 * h * w]
    if not candidates:
        return None
    return order_points(max(candidates, key=lambda c: c[0])[1])


def estimate_skew(quad: np.ndarray) -> Tuple[float, float]:
    """Trả (góc xoay mặt phẳng, góc lệch phối cảnh xấp xỉ) theo độ.
    Góc phối cảnh ước lượng từ tỉ lệ chiều cao cạnh trái/phải và trên/dưới: theta ~ arccos(ngắn/dài).
    [Nhận định] Đây là chỉ báo gần đúng để cảnh báo vượt điều kiện thiết kế, không thay cho hiệu chuẩn camera.
    """
    tl, tr, br, bl = quad
    roll = math.degrees(math.atan2(((tr - tl)[1] + (br - bl)[1]) / 2, ((tr - tl)[0] + (br - bl)[0]) / 2))
    hl, hr = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    wt, wb = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    yaw = math.degrees(math.acos(max(0.0, min(1.0, min(hl, hr) / (max(hl, hr) + 1e-6)))))
    pitch = math.degrees(math.acos(max(0.0, min(1.0, min(wt, wb) / (max(wt, wb) + 1e-6)))))
    return roll, max(yaw, pitch)


def to_ocr_input(warp_bgr: np.ndarray, layout: str, cfg: dict) -> np.ndarray:
    """Chuẩn hóa ảnh biển đã nắn -> ảnh xám (ocr_height x ocr_width).
    Biển 2 hàng: cắt hàng trên / hàng dưới rồi ghép ngang thành 1 dòng để mô hình CTC đọc trái->phải.
    """
    H, W = int(cfg.get("ocr_height", 48)), int(cfg.get("ocr_width", 192))
    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY) if warp_bgr.ndim == 3 else warp_bgr
    if layout == "two_line":
        ov = float(cfg.get("two_line_split_overlap", 0.06))
        h = gray.shape[0]
        top = gray[: int(h * (0.5 + ov))]
        bot = gray[int(h * (0.5 - ov)):]
        half = W // 2
        gray = np.hstack([cv2.resize(top, (half, H)), cv2.resize(bot, (W - half, H))])
    else:
        gray = cv2.resize(gray, (W, H))
    return gray


def rectify_plate(img: np.ndarray, box, keypoints, cfg_rect: dict, pad_ratio: float = 0.1,
                  force_layout: Optional[str] = None) -> dict:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    cx2, cy2 = int(min(W, x2 + px)), int(min(H, y2 + py))
    crop = img[cy1:cy2, cx1:cx2]

    method = "bbox"
    quad = None
    if keypoints is not None and np.isfinite(keypoints).all():
        quad, method = order_points(keypoints), "keypoints"
    else:
        q = find_quad(crop) if crop.size else None
        if q is not None:
            quad, method = q + np.array([cx1, cy1], dtype=np.float32), "contour"
    if quad is None:
        quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    tl, tr, br, bl = quad
    pw = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    ph = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    ratio = pw / (ph + 1e-6)
    layout = "one_line" if ratio >= float(cfg_rect.get("one_line_min_ratio", 2.6)) else "two_line"
    if force_layout in ("one_line", "two_line"):
        layout = force_layout
    tw, th = (240, 60) if layout == "one_line" else (160, 120)
    M = cv2.getPerspectiveTransform(quad, np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]],
                                                   dtype=np.float32))
    warp = cv2.warpPerspective(img, M, (tw, th), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    roll, persp = estimate_skew(quad)
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "warp": warp,
        "ocr_input": to_ocr_input(warp, layout, cfg_rect),
        "layout": layout,
        "quad": quad,
        "method": method,
        "aspect": float(ratio),
        "plate_height_px": float(ph),
        "sharpness": sharp,
        "skew_deg": float(max(abs(roll), persp)),
    }
