"""Dịch vụ REST nhận diện biển số cho phần mềm quản lý bãi đỗ (chế độ chụp theo trigger).

Luồng điển hình tại làn: cảm biến vòng từ/IR phát hiện xe -> bộ điều khiển làn chụp 1–3 ảnh
-> POST /v1/recognize -> nhận biển số + trạng thái -> phần mềm bãi đỗ quyết định mở barie/trừ tiền.

Chạy:  ANPR_CONFIG=config.yaml uvicorn service.app:app --host 0.0.0.0 --port 8080 --workers 1
Biến môi trường:
  ANPR_CONFIG           đường dẫn config.yaml (mặc định config.yaml)
  ANPR_API_KEY          nếu đặt, mọi request phải có header X-API-Key trùng giá trị
  ANPR_MAX_CONCURRENCY  số request suy luận đồng thời (mặc định 2)
  ANPR_MODEL_VERSION    nhãn phiên bản mô hình ghi vào kết quả (truy vết khi tra soát)
"""
from __future__ import annotations
import base64, os, threading, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
import tempfile

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from anpr import __version__
from anpr.pipeline import ANPRPipeline, load_config, _read_to_dict

import json as _json, logging
log = logging.getLogger("anpr.api")

MAX_IMAGES = 5
MAX_BYTES = 8 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024


class RecognizeJSON(BaseModel):
    images_b64: List[str] = Field(..., min_length=1, max_length=MAX_IMAGES)
    lane_id: str = ""
    direction: str = ""            # in | out
    event_id: Optional[str] = None
    return_evidence: bool = False


def _decode(data: bytes) -> np.ndarray:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Ảnh vượt quá 8 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không giải mã được ảnh")
    return img


def _jpg_b64(img) -> Optional[str]:
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def create_app(pipeline: Optional[ANPRPipeline] = None) -> FastAPI:
    state = {"pipe": pipeline}
    sem = threading.Semaphore(int(os.getenv("ANPR_MAX_CONCURRENCY", "2")))
    model_version = os.getenv("ANPR_MODEL_VERSION", "chua-gan-nhan")
    # Bộ đếm Prometheus tối giản (không phụ thuộc thư viện ngoài)
    metrics = {"requests": 0, "errors": 0, "ACCEPT": 0, "UNREAD": 0, "NO_PLATE": 0, "lat_sum_ms": 0.0}
    buckets = [50, 100, 200, 300, 500, 1000, 2000]
    hist = {b: 0 for b in buckets}
    mlock = threading.Lock()

    def pipe() -> ANPRPipeline:
        if state["pipe"] is None:
            state["pipe"] = ANPRPipeline(load_config(os.getenv("ANPR_CONFIG", "config.yaml")))
        return state["pipe"]

    def auth(x_api_key: Optional[str] = Header(default=None)):
        key = os.getenv("ANPR_API_KEY")
        if key and x_api_key != key:
            raise HTTPException(401, "Sai hoặc thiếu X-API-Key")

    @asynccontextmanager
    async def lifespan(_app):
        if os.getenv("ANPR_LAZY_LOAD") != "1":
            try:
                pipe()                                 # nạp mô hình ngay khi khởi động
            except Exception as e:                     # noqa: BLE001 – chưa có mô hình: vẫn chạy, /ready = 503
                log.error("Chua nap duoc mo hinh: %s", e)
        yield

    app = FastAPI(title="VN-ANPR Service", version=__version__, lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def test_ui():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": state["pipe"] is not None}

    @app.get("/ready")
    def ready():
        """Readiness cho bộ cân bằng tải: chỉ nhận tải khi mô hình đã nạp."""
        if state["pipe"] is None:
            try:
                pipe()                                 # thử nạp lại (mô hình vừa được chép vào volume)
            except Exception as e:                     # noqa: BLE001
                raise HTTPException(503, f"Mô hình chưa sẵn sàng: {e}")
        return {"status": "ready"}

    @app.get("/metrics")
    def prom():
        from fastapi.responses import PlainTextResponse
        with mlock:
            lines = [f'anpr_requests_total {metrics["requests"]}', f'anpr_errors_total {metrics["errors"]}']
            lines += [f'anpr_result_total{{status="{k}"}} {metrics[k]}' for k in ("ACCEPT", "UNREAD", "NO_PLATE")]
            cum = 0
            for b in buckets:
                cum += hist[b]
                lines.append(f'anpr_latency_ms_bucket{{le="{b}"}} {cum}')
            lines.append(f'anpr_latency_ms_bucket{{le="+Inf"}} {metrics["requests"]}')
            lines.append(f'anpr_latency_ms_sum {metrics["lat_sum_ms"]:.1f}')
            lines.append(f'anpr_latency_ms_count {metrics["requests"]}')
        return PlainTextResponse("\n".join(lines) + "\n")

    def _record(status: str, ms: float, err: bool = False):
        with mlock:
            metrics["requests"] += 1
            metrics["errors"] += int(err)
            if status in metrics:
                metrics[status] += 1
            metrics["lat_sum_ms"] += ms
            for b in buckets:
                if ms <= b:
                    hist[b] += 1; break

    @app.get("/v1/info", dependencies=[Depends(auth)])
    def info():
        c = pipe().cfg
        return {"service_version": __version__, "model_version": model_version,
                "detector": c["detector"].get("weights"), "ocr": c["ocr"].get("weights"),
                "accept_conf": c.get("decision", {}).get("accept_conf")}

    def _run(images, lane_id, direction, event_id, return_evidence):
        t0 = time.perf_counter()
        with sem:
            p = pipe()
            all_reads = []
            for i, img in enumerate(images):
                all_reads += p.read_frame(img, i, f"{lane_id}:{i}")
        best = max(all_reads, key=lambda r: r.quality * r.conf) if all_reads else None
        if all_reads:
            d = p.decide_reads(all_reads, key=event_id)
            status, plate = d["status"], d["text"]
        else:
            d, status, plate = {}, "NO_PLATE", ""
        out = {"event_id": event_id or str(uuid.uuid4()), "lane_id": lane_id, "direction": direction,
               "plate": plate, "status": status, "candidate": d.get("candidate", ""),
               "conf": d.get("conf", 0.0), "agree": d.get("agree", 0.0), "layout": d.get("layout", ""),
               "pattern": d.get("pattern", ""), "n_images": len(images),
               "reads": [_read_to_dict(r) for r in all_reads],
               "processing_ms": round((time.perf_counter() - t0) * 1000, 1),
               "model_version": model_version}
        if return_evidence and best is not None:
            out["evidence_jpg_b64"] = _jpg_b64(best.evidence)
        _record(status, out["processing_ms"])
        log.info(_json.dumps({k: out[k] for k in ("event_id", "lane_id", "direction", "plate", "status",
                                                   "conf", "processing_ms", "model_version")}, ensure_ascii=False))
        return out

    @app.post("/v1/recognize", dependencies=[Depends(auth)])
    def recognize(files: List[UploadFile] = File(...), lane_id: str = Form(""), direction: str = Form(""),
                  event_id: Optional[str] = Form(None), return_evidence: bool = Form(False)):
        if not 1 <= len(files) <= MAX_IMAGES:
            raise HTTPException(400, f"Gửi từ 1 đến {MAX_IMAGES} ảnh của cùng một xe")
        images = [_decode(f.file.read()) for f in files]
        return _run(images, lane_id, direction, event_id, return_evidence)

    @app.post("/v1/recognize/json", dependencies=[Depends(auth)])
    def recognize_json(body: RecognizeJSON):
        images = [_decode(base64.b64decode(b)) for b in body.images_b64]
        return _run(images, body.lane_id, body.direction, body.event_id, body.return_evidence)

    @app.post("/v1/recognize/video", dependencies=[Depends(auth)])
    def recognize_video(video: UploadFile = File(...), lane_id: str = Form(""),
                        direction: str = Form(""), event_id: Optional[str] = Form(None)):
        allowed_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in allowed_suffixes:
            raise HTTPException(400, "Chỉ hỗ trợ video MP4, AVI, MOV, MKV hoặc WebM")

        temp_path = None
        t0 = time.perf_counter()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = tmp.name
                total = 0
                while True:
                    chunk = video.file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_VIDEO_BYTES:
                        raise HTTPException(413, "Video vượt quá 200 MB")
                    tmp.write(chunk)

            with sem:
                p = pipe()
                result = p.process_video(temp_path)
            vehicles = result.get("vehicles", [])
            out = {
                "event_id": event_id or str(uuid.uuid4()),
                "lane_id": lane_id,
                "direction": direction,
                "video": {"frames": result.get("frames", 0), "fps": round(result.get("fps", 0), 2)},
                "vehicles": vehicles,
                "n_vehicles": len(vehicles),
                "processing_ms": round((time.perf_counter() - t0) * 1000, 1),
                "model_version": model_version,
            }
            _record("ACCEPT" if any(v.get("status") == "ACCEPT" for v in vehicles) else "NO_PLATE",
                    out["processing_ms"])
            return out
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    return app


app = create_app()
