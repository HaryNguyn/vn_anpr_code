"""Worker luồng camera (RTSP) cho bãi đỗ: mỗi làn một luồng xử lý, tự kết nối lại,
đẩy sự kiện xe sang phần mềm bãi đỗ qua webhook HTTP; lưu hàng đợi tạm (outbox) khi backend không phản hồi.

Chạy:  python -m service.stream_worker --config config.yaml --lanes lanes.yaml
"""
from __future__ import annotations
import argparse, json, logging, os, signal, threading, time, uuid
import urllib.request
from datetime import datetime, timedelta, timezone

import cv2
import yaml

from anpr.pipeline import ANPRPipeline, load_config

VN_TZ = timezone(timedelta(hours=7))
log = logging.getLogger("anpr.worker")


class Webhook:
    def __init__(self, cfg: dict, outbox_path: str):
        self.url = os.path.expandvars(cfg.get("url", ""))
        self.header = cfg.get("api_key_header", "X-API-Key")
        self.key = os.getenv(cfg.get("api_key_env", ""), "") if cfg.get("api_key_env") else ""
        self.timeout = float(cfg.get("timeout_s", 3))
        self.retries = int(cfg.get("retries", 3))
        self.outbox = outbox_path
        self.lock = threading.Lock()

    def send(self, event: dict) -> bool:
        data = json.dumps(event, ensure_ascii=False).encode("utf-8")
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, data=data, method="POST",
                                             headers={"Content-Type": "application/json",
                                                      **({self.header: self.key} if self.key else {})})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if 200 <= r.status < 300:
                        return True
            except Exception as e:                      # noqa: BLE001
                log.warning("Webhook lỗi lần %d: %s", attempt + 1, e)
                time.sleep(0.5 * (2 ** attempt))
        with self.lock:                                 # không mất sự kiện: ghi outbox để gửi lại
            with open(self.outbox, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return False

    def resend_outbox(self):
        with self.lock:
            if not os.path.exists(self.outbox):
                return
            lines = open(self.outbox, encoding="utf-8").read().splitlines()
            os.remove(self.outbox)
        for ln in lines:
            self.send(json.loads(ln))


class LaneWorker(threading.Thread):
    def __init__(self, lane: dict, pipe: ANPRPipeline, hook: Webhook, evidence_dir: str,
                 stop: threading.Event, model_version: str):
        super().__init__(daemon=True, name=lane["lane_id"])
        self.lane, self.pipe, self.hook, self.dir, self.stop = lane, pipe, hook, evidence_dir, stop
        self.src = os.path.expandvars(str(lane["source"]))
        self.model_version = model_version
        self.stride = max(1, int(pipe.cfg.get("video", {}).get("stride", 1)))

    def run(self):
        backoff = 1.0
        while not self.stop.is_set():
            cap = cv2.VideoCapture(self.src)
            if not cap.isOpened():
                log.error("[%s] Không mở được nguồn, thử lại sau %.0fs", self.lane["lane_id"], backoff)
                time.sleep(backoff); backoff = min(backoff * 2, 30); continue
            backoff = 1.0
            sess = self.pipe.new_session(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            idx = 0
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log.warning("[%s] Mất tín hiệu, kết nối lại", self.lane["lane_id"]); break
                if idx % self.stride == 0:
                    done, _ = sess.feed(frame, idx, self.lane["lane_id"])
                    for res, ev in done:
                        self.emit(res, ev)
                idx += 1
            for res, ev in sess.flush():
                self.emit(res, ev)
            cap.release()
            if self.lane.get("loop_file") is False and os.path.isfile(self.src):
                break                                   # nguồn là file kiểm thử: dừng khi hết

    def emit(self, res: dict, evidence):
        now = datetime.now(VN_TZ)
        eid = str(uuid.uuid4())
        path = None
        if evidence is not None and self.dir:
            day = os.path.join(self.dir, self.lane["lane_id"], now.strftime("%Y%m%d"))
            os.makedirs(day, exist_ok=True)
            path = os.path.join(day, f"{now.strftime('%H%M%S')}_{eid[:8]}.jpg")
            cv2.imwrite(path, evidence)
        event = {"event_id": eid, "event_type": "VEHICLE_PLATE", "lane_id": self.lane["lane_id"],
                 "direction": self.lane.get("direction", ""), "timestamp": now.isoformat(),
                 "plate": res["text"], "status": res["status"], "candidate": res["candidate"],
                 "conf": res["conf"], "agree": res.get("agree"), "support": res["support"],
                 "n_reads": res["n_reads"], "layout": res["layout"], "pattern": res["pattern"],
                 "evidence_path": path, "model_version": self.model_version}
        ok = self.hook.send(event)
        log.info("[%s] %s %s conf=%.3f webhook=%s", event["lane_id"], event["status"],
                 event["plate"] or event["candidate"], event["conf"], "OK" if ok else "OUTBOX")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--lanes", default="lanes.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lanes_cfg = yaml.safe_load(open(a.lanes, encoding="utf-8"))
    pipe = ANPRPipeline(load_config(a.config))
    os.makedirs(lanes_cfg.get("work_dir", "runtime"), exist_ok=True)
    hook = Webhook(lanes_cfg["webhook"], os.path.join(lanes_cfg.get("work_dir", "runtime"), "outbox.jsonl"))
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    mv = os.getenv("ANPR_MODEL_VERSION", "chua-gan-nhan")
    workers = [LaneWorker(l, pipe, hook, lanes_cfg.get("evidence_dir", ""), stop, mv) for l in lanes_cfg["lanes"]]
    for w in workers:
        w.start()
    while not stop.is_set() and any(w.is_alive() for w in workers):
        stop.wait(30)
        hook.resend_outbox()
    for w in workers:
        w.join(timeout=5)


if __name__ == "__main__":
    main()
