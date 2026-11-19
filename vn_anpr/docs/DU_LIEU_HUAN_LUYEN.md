# Chuẩn bị dữ liệu huấn luyện

## 1. Nguồn dữ liệu thật

- VNLP: https://github.com/fict-labs/VNLP
  Theo mô tả của tác giả: khoảng 37.300 ảnh, có biển một hàng/hai hàng,
  nhãn vị trí và chuỗi ký tự. Xem link tải trong README của nguồn.
  Cần kiểm tra cấu trúc archive, phạm vi ảnh (ảnh toàn cảnh hay crop),
  điều kiện sử dụng và split gốc trước khi chuyển đổi.
- Detector: https://universe.roboflow.com/o23ps6a/license_plates_vietnam/dataset/3
  Trang công bố 4.534 ảnh, giấy phép CC BY 4.0; có định dạng YOLO.
  Nhãn hộp biển số không đồng nghĩa với nhãn chuỗi OCR.

Các nguồn trên mới được tìm kiếm, chưa tải và kiểm định trong dự án.
Ưu tiên bổ sung ảnh camera triển khai thực tế. Gán hộp biển số và chuỗi ký tự
bằng CVAT hoặc Label Studio, sau đó kiểm tra thủ công.
Chia train/val/test theo camera, ngày và xe; không để các khung hình gần
trùng của một xe xuất hiện ở nhiều tập. Giữ test thực độc lập.

## 2. Dữ liệu tổng hợp OCR

Chạy từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe scripts\generate_synthetic_ocr.py --count 10000 --out data\synthetic_ocr_10k
```

Script tạo ảnh crop biển một/hai hàng, nền trắng/vàng, thay đổi độ sáng,
xoay nhẹ, mờ và nhiễu; xuất `ocr_train.csv`, `ocr_val.csv` có các cột
`path,label,layout`. Đường dẫn ảnh tương đối với CSV, đúng với
`train_ocr_crnn.py`. Chuỗi nhãn không trùng giữa train và val.
Thư mục có sẵn sẽ không bị ghi đè.

Đây là dữ liệu khởi động để thử pipeline hoặc tiền huấn luyện OCR.
Font Hershey không phải font biển thật; không dùng độ chính xác trên
validation tổng hợp để khẳng định khả năng nhận diện thực tế.
Dữ liệu này không đủ để huấn luyện detector trên cảnh xe thật.

```powershell
.\.venv\Scripts\python.exe -m pip install torch
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe train_ocr_crnn.py --train data/synthetic_ocr/ocr_train.csv --val data/synthetic_ocr/ocr_val.csv --epochs 80 --out models
```

Lệnh trên dùng bộ khởi động mặc định `data/synthetic_ocr`; đổi đường dẫn
nếu sinh bộ 10k. Huấn luyện xong cần fine-tune bằng crop thật có nhãn.
Chạy script sinh dữ liệu không tự tạo model ONNX.

## 3. Dữ liệu detector

Xuất một lớp `plate` dưới định dạng YOLO:

```text
data/plates/
  images/train/...
  images/val/...
  images/test/...
  labels/train/...
  labels/val/...
  labels/test/...
  data.yaml
```

Mỗi dòng nhãn: `0 cx cy width height`, tọa độ chuẩn hóa theo kích thước ảnh.
`data.yaml` cần khai báo `path`, `train`, `val`, `test`, `names: {0: plate}`.
Script huấn luyện hiện tại sử dụng cả split `test`.

```powershell
.\.venv\Scripts\python.exe -m pip install ultralytics
.\.venv\Scripts\python.exe train_detector.py --data data/plates/data.yaml --model yolo11s.pt --device cpu
```

Nếu có CUDA, chọn `--device 0`. Sau huấn luyện, sao chép ONNX xuất ra thành
`models/plate_det.onnx`. OCR xuất `models/plate_ocr.onnx`.
Detector bbox dùng `num_keypoints: 0`; charset OCR phải khớp config lúc train.
