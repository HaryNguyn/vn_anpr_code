"""Kiểm thử dịch vụ REST và worker RTSP (dùng mô hình giả lập, webhook HTTP cục bộ)."""
import base64, json, os, sys, threading, tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_core import CFG, MockDet, MockOCR, place, make_plate
from anpr.pipeline import ANPRPipeline


def test_rest_api():
    from fastapi.testclient import TestClient
    from service.app import create_app
    os.environ["ANPR_API_KEY"] = "k"
    img, box, _ = place(make_plate(False), yaw_deg=20)
    app = create_app(ANPRPipeline(CFG, MockDet(box), MockOCR("30A12345")))
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    jpg = cv2.imencode(".jpg", img)[1].tobytes()
    assert c.post("/v1/recognize", files=[("files", ("a.jpg", jpg, "image/jpeg"))]).status_code == 401
    r = c.post("/v1/recognize", headers={"X-API-Key": "k"},
               files=[("files", ("a.jpg", jpg, "image/jpeg")), ("files", ("b.jpg", jpg, "image/jpeg"))],
               data={"lane_id": "CONG_A_VAO", "direction": "in", "event_id": "e1", "return_evidence": "true"})
    j = r.json()
    assert r.status_code == 200 and j["plate"] == "30A12345" and j["status"] == "ACCEPT", j
    assert j["event_id"] == "e1" and j["n_images"] == 2 and j["evidence_jpg_b64"]
    r2 = c.post("/v1/recognize/json", headers={"X-API-Key": "k"},
                json={"images_b64": [base64.b64encode(jpg).decode()], "lane_id": "L1"})
    assert r2.json()["plate"] == "30A12345"
    m = c.get("/metrics").text
    assert 'anpr_result_total{status="ACCEPT"} 2' in m and c.get("/ready").status_code == 200
    print("REST:", {k: j[k] for k in ("plate", "status", "conf", "processing_ms")})


def test_stream_worker():
    from service.stream_worker import LaneWorker, Webhook
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200); self.end_headers()
        def log_message(self, *a): pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as d:
        vp = os.path.join(d, "v.mp4")
        wr = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
        boxes = []
        for i in range(10):
            fr, b, _ = place(make_plate(False), yaw_deg=20, center=(500 + 10 * i, 400)); wr.write(fr); boxes.append(b)
        wr.release()

        class SeqDet:
            i = 0
            def __call__(self, img):
                b = boxes[min(self.i, 9)]; self.i += 1
                from anpr.types import Detection
                return [Detection(b, .9, 0, None)]
        pipe = ANPRPipeline(CFG, SeqDet(), MockOCR("29B112345"))
        hook = Webhook({"url": f"http://127.0.0.1:{srv.server_port}/ev"}, os.path.join(d, "outbox.jsonl"))
        w = LaneWorker({"lane_id": "CONG_A_VAO", "direction": "in", "source": vp, "loop_file": False},
                       pipe, hook, os.path.join(d, "ev"), threading.Event(), "test")
        w.start(); w.join(20)
        assert got and got[0]["plate"] == "29B112345" and got[0]["status"] == "ACCEPT", got
        assert os.path.exists(got[0]["evidence_path"])
        print("WEBHOOK:", {k: got[0][k] for k in ("lane_id", "plate", "status", "support", "n_reads")})
    srv.shutdown()


if __name__ == "__main__":
    test_rest_api(); print("PASS test_rest_api")
    test_stream_worker(); print("PASS test_stream_worker")
