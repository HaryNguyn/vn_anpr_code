# VN-ANPR – Nhận diện biển số ô tô, xe máy Việt Nam từ ảnh và video

Phiên bản 1.0 – khung phần mềm sẵn sàng tích hợp mô hình huấn luyện trên dữ liệu biển Việt Nam.

## 1. Phạm vi và cam kết kỹ thuật
| Hạng mục | Nội dung |
|---|---|
| Đầu vào | Nhiều ảnh (thư mục, có thể nhóm theo xe) hoặc video/RTSP |
| Điều kiện thiết kế | Góc lệch so với phương vuông góc biển ≤ 30°, ánh sáng đủ |
| Loại biển | Ô tô 1 hàng (biển dài), ô tô 2 hàng (biển ngắn), xe máy 2 hàng 190×140 mm |
| Mục tiêu | ≥ 95% biển đọc đúng hoàn toàn (exact match, mức biển), chứng minh bằng cận dưới CI 95% |
| Không cam kết | Ban đêm không IR, mưa lớn, biển bẩn/che, xe tốc độ cao không trigger – ngoài điều kiện thiết kế |

**Quan trọng:** độ chính xác 95% là thuộc tính của *mô hình + dữ liệu + camera*, không phải của riêng đoạn mã.
Mã nguồn này cung cấp pipeline tối ưu, công cụ huấn luyện và công cụ nghiệm thu; độ chính xác chỉ được
công bố sau khi chạy `evaluate.py` trên tập nghiệm thu độc lập.

## 2. Kiến trúc
```
Ảnh/khung hình -> [1] Phát hiện biển (YOLO ONNX, hộp bao hoặc 4 góc)
               -> [2] Nắn phối cảnh (homography; 4 góc > tứ giác đường viền > hộp bao)
               -> [3] Phân loại bố cục 1 hàng / 2 hàng (tỷ lệ W/H) -> biển 2 hàng tách hàng & ghép ngang
               -> [4] OCR không phân đoạn CRNN + CTC (top-k từng ký tự)
               -> [5] Văn phạm TT 79/2024/TT-BCA: khớp / tìm tổ hợp top-k hợp lệ / sửa nhầm theo vị trí
               -> [6] Kiểm soát chất lượng: chiều cao biển, độ nét, góc lệch -> ACCEPT | UNREAD | LOW_QUALITY
               -> [7] Video: theo vết + bỏ phiếu đa khung hình có trọng số (conf x chất lượng)
```
| File | Vai trò |
|---|---|
| `anpr/detector.py` | Phát hiện biển (onnxruntime hoặc ultralytics), NMS, giải mã 4 góc |
| `anpr/rectify.py` | Nắn chỉnh, ước lượng góc lệch, chuẩn hóa đầu vào OCR (dùng chung train/infer) |
| `anpr/ocr.py` | Giải mã CTC, độ tin cậy từng ký tự, top-k |
| `anpr/grammar.py` | Văn phạm biển số Việt Nam, sửa lỗi có kiểm soát |
| `anpr/tracker.py` | Theo vết IoU, bỏ phiếu đa khung hình |
| `anpr/pipeline.py` | Ghép nối, xử lý ảnh/video, quyết định chấp nhận |
| `run_anpr.py` | Dòng lệnh chạy |
| `evaluate.py` | Nghiệm thu: exact-match, CER, Wilson CI 95%, accuracy-coverage |
| `train_detector.py`, `train_ocr_crnn.py` | Huấn luyện và export ONNX |
| `tests/test_core.py` | Kiểm thử logic bằng dữ liệu tổng hợp |

## 3. Cài đặt và chạy
```bash
pip install -r requirements.txt
# đặt models/plate_det.onnx và models/plate_ocr.onnx, chỉnh config.yaml
python run_anpr.py --images data/anh/ --out ketqua/
python run_anpr.py --images data/theo_xe/ --group-by-folder --out ketqua/   # nhiều ảnh cùng 1 xe
python run_anpr.py --video cong_vao.mp4 --save-video --out ketqua/
python run_anpr.py --images data/raw/ --save-crops data/crops/            # thu crop để gán nhãn
python evaluate.py --gt data/test_gt.csv --pred ketqua/ket_qua_anh.csv --target 0.95
python tests/test_core.py
```

trang test :
```bash
.\.venv\Scripts\Activate.ps1
$env:ANPR_CONFIG = "config.yaml"
python -m uvicorn service.app:app --host 0.0.0.0 --port 8080
```

## 4. Dữ liệu và huấn luyện (điều kiện cần để đạt ≥ 95%)
- Detector: khuyến nghị **YOLO-pose 4 góc** (`num_keypoints: 4`) để nắn chính xác ở góc lệch 20–30°.
- OCR: crop biển gán nhãn chuỗi; cân bằng ô tô/xe máy, 1 hàng/2 hàng, biển trắng/vàng/xanh, tem nhận diện.
- Chia tập theo **trạm/camera và ngày**, không chia ngẫu nhiên theo ảnh (tránh rò rỉ ảnh gần trùng).
- Cỡ tập nghiệm thu: ~1.825 biển cho mỗi nhóm báo cáo để ước lượng 95% với sai số ±1% (công thức trong `evaluate.py`).
- Báo cáo tách: ô tô/xe máy, 1 hàng/2 hàng, dải góc 0–15° và 15–30°.
- Định lượng cụ thể số ảnh huấn luyện cần thiết: **chưa có sở cứ chính thống** – xác định bằng đường cong học (learning curve) trên dữ liệu VDTC.

## 4b. Chạy như module tích hợp bãi đỗ
```bash
pip install ".[service]"
ANPR_CONFIG=config.yaml ANPR_API_KEY=xxx uvicorn service.app:app --host 0.0.0.0 --port 8080
python -m service.stream_worker --config config.yaml --lanes lanes.yaml
docker compose up -d --build        # cần file .env (xem .env.example)
```
Chi tiết hợp đồng API, sự kiện webhook, trình tự tại làn: `docs/TICH_HOP_BAI_DO.md`.

Windows / Docker Windows containers / OpenStack: `docs/TRIEN_KHAI_WINDOWS_OPENSTACK.md`,
`Dockerfile.windows`, `scripts/windows/*.ps1`, `deploy/openstack/*.yaml`.

## 5. Bản quyền thành phần
- `ultralytics` (YOLO) phát hành theo AGPL-3.0; dùng trong sản phẩm thương mại đóng mã cần Enterprise License của Ultralytics.
  Pipeline suy luận chỉ phụ thuộc onnxruntime và chuẩn đầu ra ONNX, nên có thể thay detector bằng mô hình có giấy phép
  thương mại phù hợp; cần bộ phận pháp chế xác nhận trước khi thương mại hóa.
- Mã nguồn trong thư mục này do VDTC sở hữu khi đưa vào sử dụng (không dùng mã bên thứ ba ngoài thư viện liệt kê).

## 6. Hạn chế đã biết
- Văn phạm: các mẫu đánh dấu [CẦN KIỂM CHỨNG] trong `config.yaml` phải đối chiếu Phụ lục 02–04 TT 79/2024/TT-BCA;
  biển quân sự/công an đặc thù, biển tạm in giấy chưa được mô hình hóa.
- Ước lượng góc lệch từ tứ giác là xấp xỉ, dùng để cảnh báo, không thay hiệu chuẩn camera.
- Phương án nắn bằng đường viền là dự phòng; viền biển mờ hoặc khung biển trang trí làm giảm độ chính xác.
- Bộ theo vết IoU phù hợp cổng/barie/bãi đỗ; cao tốc, MLFF nên thay ByteTrack/OC-SORT có Kalman.
