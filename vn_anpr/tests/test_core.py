"""Kiểm thử logic lõi bằng dữ liệu tổng hợp (không cần trọng số mô hình thật).
Chạy: python -m pytest -q tests/   hoặc   python tests/test_core.py
LƯU Ý: kiểm thử này xác nhận ĐÚNG LOGIC phần mềm, KHÔNG chứng minh độ chính xác 95% – việc đó cần
mô hình huấn luyện trên dữ liệu biển Việt Nam và tập nghiệm thu độc lập (evaluate.py).
"""
import math, os, sys, csv, subprocess, tempfile
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anpr.pipeline import ANPRPipeline, load_config
from anpr.rectify import rectify_plate
from anpr.grammar import PlateGrammar
from anpr.ocr import ctc_greedy_decode
from anpr.types import Detection, OcrResult
from anpr.tracker import IoUTracker, vote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_config(os.path.join(ROOT, "config.yaml"))
CHARSET = CFG["ocr"]["charset"]


def make_plate(two_line=False):
    if two_line:   # xe máy 190x140 mm (QCVN 08:2024/BCA) phóng x2
        p = np.full((280, 380, 3), 255, np.uint8)
        cv2.rectangle(p, (4, 4), (375, 275), (0, 0, 0), 6)
        cv2.putText(p, "29-B1", (70, 115), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 9)
        cv2.putText(p, "123.45", (35, 240), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 9)
    else:          # biển dài ô tô
        p = np.full((110, 520, 3), 255, np.uint8)
        cv2.rectangle(p, (3, 3), (516, 106), (0, 0, 0), 5)
        cv2.putText(p, "30A-123.45", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.1, (0, 0, 0), 6)
    return p


def place(plate, yaw_deg=25, roll_deg=5, scale=0.5, center=(640, 400)):
    """Đặt biển vào khung 1280x720 với góc lệch phối cảnh (yaw) và xoay mặt phẳng (roll)."""
    bg = np.random.default_rng(0).integers(60, 140, (720, 1280, 3), dtype=np.uint8)
    h, w = plate.shape[:2]
    W, H = w * scale, h * scale
    k = math.cos(math.radians(yaw_deg))
    quad = np.float32([[-W / 2, -H / 2], [W / 2 * k, -H / 2 * k], [W / 2 * k, H / 2 * k], [-W / 2, H / 2]])
    a = math.radians(roll_deg)
    R = np.float32([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    quad = quad @ R.T + np.float32(center)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad)
    warped = cv2.warpPerspective(plate, M, (1280, 720))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (1280, 720))
    bg[mask > 0] = warped[mask > 0]
    x1, y1 = quad.min(0); x2, y2 = quad.max(0)
    return bg, (float(x1), float(y1), float(x2), float(y2)), quad


class MockDet:
    def __init__(self, box): self.box = box
    def __call__(self, img): return [Detection(self.box, 0.9, 0, None)]


class MockOCR:
    def __init__(self, text, conf=0.97): self.text, self.conf = text, conf
    def __call__(self, gray):
        assert gray.shape == (48, 192)
        return OcrResult(self.text, self.conf, [self.conf] * len(self.text),
                         [[(c, self.conf)] for c in self.text])


def test_rectify_one_and_two_line():
    img, box, _ = place(make_plate(False), yaw_deg=28)
    r = rectify_plate(img, box, None, CFG["rectify"])
    assert r["method"] == "contour" and r["layout"] == "one_line", (r["method"], r["layout"])
    assert 15 <= r["skew_deg"] <= 40, r["skew_deg"]
    img, box, _ = place(make_plate(True), yaw_deg=25, scale=0.35)
    r = rectify_plate(img, box, None, CFG["rectify"])
    assert r["layout"] == "two_line" and r["ocr_input"].shape == (48, 192)


def test_grammar():
    g = PlateGrammar(CFG["grammar"])
    assert g.match("30A12345") and g.match("29B112345") and g.match("29MĐ112345")
    assert not g.match("30A1234567")
    # top-k: vị trí 2 mô hình đọc 'O' nhưng phương án 2 là '0'
    alts = [[(c, .99)] for c in "3"] + [[("O", .55), ("0", .44)]] + [[(c, .99)] for c in "A12345"]
    t, name, conf = g.correct(OcrResult("3OA12345", .55, [], alts))
    assert t == "30A12345" and name, (t, name)
    t, name, _ = g.correct(OcrResult("3OA12345", .9))          # sửa theo vị trí
    assert t == "30A12345", t


def test_ctc_decode():
    C = len(CHARSET) + 1
    seq = "30A12345"
    T = 2 * len(seq) + 1
    probs = np.full((T, C), 1e-4)
    for t in range(T):
        if t % 2 == 1:
            probs[t, CHARSET.index(seq[t // 2]) + 1] = 0.98
        else:
            probs[t, 0] = 0.98
    probs /= probs.sum(1, keepdims=True)
    r = ctc_greedy_decode(probs, CHARSET)
    assert r.text == seq and r.conf > 0.9


def test_vote():
    from anpr.types import PlateRead
    mk = lambda t, c: PlateRead((0, 0, 10, 10), .9, "one_line", t, t, c, True, "p", 1.0, 40, 100, 5, "ACCEPT")
    v = vote([mk("30A12345", .95), mk("30A12345", .93), mk("30A12346", .60), mk("30A12345", .9)])
    assert v["text"] == "30A12345" and v["support"] == 3 and v["conf"] > 0.7


def test_pipeline_images_and_video():
    img, box, _ = place(make_plate(False), yaw_deg=25)
    pipe = ANPRPipeline(CFG, MockDet(box), MockOCR("30A12345"))
    reads = pipe.read_frame(img, 0, "x.jpg")
    assert reads[0].status == "ACCEPT" and reads[0].text == "30A12345"
    with tempfile.TemporaryDirectory() as d:
        vp = os.path.join(d, "v.mp4")
        wr = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
        boxes = []
        for i in range(12):
            fr, b, _ = place(make_plate(False), yaw_deg=25, center=(500 + 10 * i, 400))
            wr.write(fr); boxes.append(b)
        wr.release()

        class SeqDet:
            i = 0
            def __call__(self, img):
                b = boxes[min(self.i, len(boxes) - 1)]; self.i += 1
                return [Detection(b, .9, 0, None)]
        texts = iter(["30A12345"] * 9 + ["30A12846", "3OA12345", "30A12345"])

        class SeqOCR:
            def __call__(self, gray):
                t = next(texts, "30A12345")
                return OcrResult(t, .95, [.95] * len(t), [[(c, .95)] for c in t])
        res = ANPRPipeline(CFG, SeqDet(), SeqOCR()).process_video(vp)
        assert len(res["vehicles"]) == 1, res
        assert res["vehicles"][0]["text"] == "30A12345", res["vehicles"][0]


def _const_model(path, out_arr, in_shape):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    inp = helper.make_tensor_value_info("image", TensorProto.FLOAT, list(in_shape))
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, None)
    nodes = [helper.make_node("ReduceSum", ["image"], ["s"], keepdims=1),
             helper.make_node("Mul", ["s", "zero"], ["z"]),
             helper.make_node("Add", ["z", "const"], ["out"])]
    g = helper.make_graph(nodes, "g", [inp], [out],
                          [numpy_helper.from_array(np.float32([0]).reshape(1, 1, 1, 1), "zero"),
                           numpy_helper.from_array(out_arr.astype(np.float32), "const")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
    onnx.save(m, path)


def test_onnx_backends():
    try:
        import onnx  # noqa
    except ImportError:
        print("bỏ qua test ONNX (chưa cài onnx)"); return
    from anpr.detector import OnnxYoloDetector
    from anpr.ocr import OnnxCtcRecognizer
    with tempfile.TemporaryDirectory() as d:
        # detector: ảnh 1280x720 -> letterbox 640 (r=0.5, top=140); 1 hộp tại (600,380)-(760,420) ảnh gốc
        N = 8; out = np.zeros((1, 5, N), np.float32)
        out[0, :, 0] = [340, 140 + 200, 80, 20, 0.9]
        out[0, :, 1] = [342, 140 + 201, 78, 19, 0.6]      # trùng -> NMS loại
        _const_model(os.path.join(d, "det.onnx"), out, (1, 3, 640, 640))
        det = OnnxYoloDetector({"weights": os.path.join(d, "det.onnx"), "input_size": 640, "num_classes": 1})
        dets = det(np.zeros((720, 1280, 3), np.uint8))
        assert len(dets) == 1 and abs(dets[0].box[0] - 600) < 1 and abs(dets[0].box[3] - 420) < 1, dets
        # OCR: logit cho "30A12345", đầu ra (T,1,C)
        C = len(CHARSET) + 1; seq = "30A12345"; T = 2 * len(seq) + 1
        lg = np.full((T, 1, C), -5.0)
        for t in range(T):
            lg[t, 0, CHARSET.index(seq[t // 2]) + 1 if t % 2 else 0] = 5.0
        _const_model(os.path.join(d, "ocr.onnx"), lg, (1, 1, 48, 192))
        ocr = OnnxCtcRecognizer({"weights": os.path.join(d, "ocr.onnx"), "charset": CHARSET})
        r = ocr(np.zeros((48, 192), np.uint8))
        assert r.text == seq and r.conf > 0.9, r


def test_evaluate_script():
    with tempfile.TemporaryDirectory() as d:
        gt, pr = os.path.join(d, "gt.csv"), os.path.join(d, "pr.csv")
        rng = np.random.default_rng(1)
        with open(gt, "w", newline="") as f1, open(pr, "w", newline="") as f2:
            w1, w2 = csv.writer(f1), csv.writer(f2)
            w1.writerow(["source", "plate", "vehicle_type"]); w2.writerow(["source", "candidate", "conf", "status"])
            for i in range(1000):
                t = f"30A{i:05d}"; w1.writerow([f"{i}.jpg", t, "oto" if i % 2 else "xe_may"])
                ok = rng.random() < 0.975
                w2.writerow([f"{i}.jpg", t if ok else "30A99999", 0.97 if ok else 0.7, "ACCEPT" if ok else "UNREAD"])
        out = subprocess.run([sys.executable, os.path.join(ROOT, "evaluate.py"), "--gt", gt, "--pred", pr],
                             capture_output=True, text=True, check=True).stdout
        assert "TOÀN BỘ" in out and "CI95%" in out
        print(out)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
