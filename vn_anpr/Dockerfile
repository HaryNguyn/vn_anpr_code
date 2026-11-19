# Ảnh triển khai CPU. GPU: thay base bằng nvidia/cuda + onnxruntime-gpu (xem docs/TICH_HOP_BAI_DO.md)
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Ho_Chi_Minh
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 ffmpeg tzdata \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY anpr ./anpr
COPY service ./service
COPY run_anpr.py evaluate.py config.yaml lanes.yaml ./
RUN pip install --no-cache-dir ".[service]"
# Mô hình KHÔNG đóng vào image: gắn thư mục /app/models khi chạy để cập nhật mô hình không phải build lại
VOLUME ["/app/models", "/app/runtime"]
RUN useradd -r -u 10001 anpr && mkdir -p /app/runtime && chown -R anpr /app/runtime
USER anpr
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health')"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
