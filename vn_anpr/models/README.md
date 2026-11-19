# Models hiện có

## plate_det.onnx

- Model pretrained công khai: https://huggingface.co/ishaqinu/yolo26n-license-plate-detection
- YOLO26n, một lớp biển số, không có NMS tích hợp.
- Input float32 `[1,3,640,640]`, output `[1,5,8400]`.
- Tương thích backend ONNX hiện tại: `input_size: 640`, `num_classes: 1`,
  `num_keypoints: 0` trong `config.yaml`.
- Nguồn, revision và SHA256: `plate_det.source.json`.
- Giấy phép AGPL-3.0; mô tả và attribution gốc được lưu trong
  `detector_upstream_README.md`, `detector_ATTRIBUTION.md`.
- Đây là model tải về, không phải detector đã huấn luyện trên dữ liệu
  Việt Nam tại dự án này. Chưa đo độ chính xác trên camera thực tế.

## plate_ocr.onnx / plate_ocr_best.pt

CRNN khởi tạo từ checkpoint tổng hợp, đã fine-tune 80 epoch trên
3.603 ảnh biển thật; validation 377 ảnh, test 399 ảnh.
Các tập được tách theo chuỗi biển số và loại trùng nội dung file.
Validation: 351/377 (93,10%); test: 364/399 (91,23%) đọc đúng toàn chuỗi.
Đây là kết quả OCR trên ảnh crop, không phải độ chính xác toàn pipeline.
Đã kiểm tra ONNX và đối chiếu đầu ra với PyTorch trước khi triển khai.
Báo cáo: `plate_ocr.training.json`; bản cũ: `backups/20261118_225948/`.
Charset và tiền xử lý phải giữ đúng `config.yaml` dùng lúc huấn luyện.

## Chạy

Từ thư mục gốc:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe run_anpr.py --images data/anh --out ketqua
```

Đã kiểm tra cấu trúc detector ONNX và chạy CLI với cả hai model trên một
ảnh tổng hợp. Đây là kiểm tra tích hợp, không phải đánh giá độ chính xác.
Để cải thiện thực tế: fine-tune detector bằng ảnh xe có nhãn bbox và
OCR bằng crop biển thật có nhãn chuỗi; đánh giá trên tập test độc lập.
