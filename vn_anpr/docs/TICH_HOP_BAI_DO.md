# Hướng dẫn tích hợp VN-ANPR vào phần mềm quản lý bãi đỗ

## 1. Ba phương thức tích hợp
| Phương thức | Khi nào dùng | Điểm vào |
|---|---|---|
| A. Thư viện nhúng (in-process) | Phần mềm bãi đỗ viết bằng Python, chạy cùng máy | `from anpr.pipeline import ANPRPipeline` |
| B. Dịch vụ REST theo trigger | Làn có vòng từ/cảm biến IR, bộ điều khiển làn chụp ảnh khi xe dừng trước barie (khuyến nghị cho bãi đỗ) | `POST /v1/recognize` |
| C. Worker luồng RTSP + webhook | Làn không có trigger, hoặc cần giám sát liên tục, đối soát hậu kiểm | `service.stream_worker` đẩy sự kiện sang backend |

## 2. Hợp đồng API (phương thức B)
`POST /v1/recognize` – multipart/form-data, header `X-API-Key`
| Trường | Kiểu | Ghi chú |
|---|---|---|
| files | 1–5 ảnh JPEG/PNG | các ảnh của CÙNG một xe, tối đa 8 MB/ảnh |
| lane_id | chuỗi | mã làn, VD CONG_A_VAO |
| direction | in / out | |
| event_id | chuỗi | mã giao dịch do phần mềm bãi đỗ sinh, dùng chống trùng và tra soát |
| return_evidence | true/false | trả ảnh crop biển base64 để lưu bằng chứng |

`POST /v1/recognize/json` – cùng nội dung, ảnh dạng `images_b64` (cho bộ điều khiển không gửi được multipart).

Phản hồi:
```json
{"event_id":"e1","lane_id":"CONG_A_VAO","direction":"in","plate":"30A12345","status":"ACCEPT",
 "candidate":"30A12345","conf":0.97,"agree":1.0,"layout":"one_line","pattern":"oto_trong_nuoc_5so",
 "n_images":2,"reads":[...],"processing_ms":85.2,"model_version":"v1.0","evidence_jpg_b64":"..."}
```
| status | Ý nghĩa | Hành động đề xuất tại làn |
|---|---|---|
| ACCEPT | Biển hợp lệ văn phạm, đủ tin cậy | Tự động: ghi vé vào / tính phí ra, mở barie theo nghiệp vụ |
| UNREAD | Có biển nhưng chưa đủ tin cậy | Phương án dự phòng: NFC/thẻ/QR/ePass, hoặc nhân viên xác nhận `candidate` |
| NO_PLATE | Không phát hiện biển | Chụp lại (trigger lần 2) rồi chuyển dự phòng |

`GET /health` (giám sát), `GET /v1/info` (phiên bản mô hình, ngưỡng).

## 3. Sự kiện webhook (phương thức C)
Worker gửi `POST` JSON tới `PARKING_WEBHOOK_URL` khi một xe rời khung hình:
```json
{"event_id":"uuid","event_type":"VEHICLE_PLATE","lane_id":"CONG_A_VAO","direction":"in",
 "timestamp":"2026-11-17T08:15:02+07:00","plate":"29B112345","status":"ACCEPT","candidate":"29B112345",
 "conf":0.95,"agree":1.0,"support":10,"n_reads":10,"layout":"two_line","pattern":"moto_trong_nuoc",
 "evidence_path":"runtime/evidence/CONG_A_VAO/20261117/081502_ab12cd34.jpg","model_version":"v1.0"}
```
Backend phải trả HTTP 2xx và xử lý trùng theo `event_id` (worker gửi lại từ outbox khi lỗi mạng).

## 4. Trình tự tại làn có barie (phương thức B)
```
Vòng từ/IR -> Bộ điều khiển làn -> chụp 2–3 ảnh cách 150–300 ms -> POST /v1/recognize
          -> ACCEPT: phần mềm bãi đỗ tạo phiên vào / tính phí ra -> lệnh mở barie
          -> UNREAD/NO_PLATE: chuyển thẻ/NFC/QR/nhân viên; lưu ảnh để gán nhãn huấn luyện lại
```
Khoảng cách giữa các ảnh là [Nhận định]; hiệu chỉnh theo tốc độ xe vào làn thực tế.

## 5. Ngân sách độ trễ cần đo khi nghiệm thu [Nhận định – chưa có số đo thực tế]
Chụp ảnh + truyền + phát hiện + OCR + quyết định nghiệp vụ. Đo p95, p99 bằng trường `processing_ms`
và log phía phần mềm bãi đỗ; đặt mục tiêu theo yêu cầu vận hành làn của VDTC.
