# Triển khai VN-ANPR dạng microservice trên Windows, Docker, OpenStack

## 1. Kiến trúc microservice
| Dịch vụ | Trạng thái | Mở rộng | Cổng / giao tiếp |
|---|---|---|---|
| `anpr-api` | Không trạng thái (stateless) | Ngang: nhiều node sau Octavia LB | HTTP 8080: `/v1/recognize`, `/health`, `/ready`, `/metrics` |
| `anpr-worker` | Có trạng thái theo làn (track đang mở, outbox) | Chia làn theo node (mỗi làn chỉ 1 worker) | RTSP vào, webhook HTTP ra |
| Kho mô hình | Tệp ONNX theo phiên bản | Volume Cinder / thư mục chia sẻ | Gắn chỉ đọc vào container |
| Ảnh bằng chứng | Dữ liệu cá nhân | Cinder (node) hoặc Swift (tập trung) | Thời hạn lưu theo quy định Pháp chế |

## 2. Hai phương án hạ tầng
| | PA1 – Windows Server 2022 + Windows containers (theo yêu cầu) | PA2 – Linux VM + Linux containers (đối chứng) |
|---|---|---|
| Ảnh container | `Dockerfile.windows` (Server Core ltsc2022) | `Dockerfile` (python slim) |
| Runtime | Docker CE/Moby, Mirantis Container Runtime hoặc containerd | Docker CE |
| Ràng buộc | Host phải là Windows Server 2022 khi chạy process isolation; Hyper-V isolation trong VM KVM cần ảo hóa lồng nhau | Ít ràng buộc |
| Kích thước ảnh | Lớn (hàng GB, gồm Server Core) | Nhỏ hơn |
| GPU | Hạn chế, phải kiểm chứng riêng | Hỗ trợ GPU passthrough/vGPU thuận lợi hơn [Nhận định] |
| Bản quyền | Cần giấy phép Windows Server cho host | Không phát sinh giấy phép HĐH |

Mã nguồn chạy được cả hai phương án, cùng hợp đồng API; chỉ khác ảnh container.

## 3. Các bước triển khai PA1 trên OpenStack
1. **Ảnh mẫu Glance:** dựng Windows Server 2022 trên KVM, cài VirtIO, QEMU guest agent, Cloudbase-Init;
   chạy `scripts/windows/install_docker_host.ps1` (máy khởi động lại); kéo sẵn `servercore:ltsc2022`; Sysprep; tải lên Glance.
2. **Build ảnh ứng dụng** trên một máy Windows Server 2022 (cùng phiên bản host):
   `scripts\windows\build_and_run_containers.ps1 -Registry registry.vdtc.local -Tag 1.0.0 -Push`
3. **Tạo node:** `openstack stack create -t deploy/openstack/heat_anpr_windows.yaml -e env_windows.yaml anpr-node-01`
   (tạo security group, port, ổ Cinder D:, chạy container qua Cloudbase-Init).
4. **Phát hành mô hình:** chép `plate_det.onnx`, `plate_ocr.onnx` vào `D:\anpr\models`, gọi `/ready` để nạp; ghi `ANPR_MODEL_VERSION`.
5. **Cân bằng tải:** khi ≥ 2 node, `openstack stack create -t deploy/openstack/heat_anpr_lb.yaml ...`; health monitor kiểm tra `/ready`.
6. **Giám sát:** thu `/metrics` (Prometheus), log JSON từng giao dịch, cảnh báo khi tỷ lệ UNREAD hoặc p95 độ trễ tăng.
7. **Nghiệm thu:** chạy `evaluate.py` trên tập ảnh của chính bãi đỗ triển khai, kết luận theo cận dưới CI 95%.

## 4. Vị trí đặt worker RTSP [Nhận định]
Đưa nhiều luồng RTSP 4 MP từ bãi đỗ về trung tâm dữ liệu OpenStack tốn băng thông và tăng độ trễ phụ thuộc đường truyền.
Khuyến nghị: làn có trigger gửi ảnh tĩnh về `anpr-api` trung tâm; worker RTSP đặt tại máy edge ở bãi (Windows native
hoặc container), chỉ gửi sự kiện và ảnh crop về trung tâm. Băng thông mỗi luồng phụ thuộc codec và bitrate camera – cần đo.

## 5. Hạng mục phải kiểm chứng trước khi chốt (chưa thử nghiệm được trong môi trường soạn thảo)
- Build `Dockerfile.windows` thành công và `import cv2` trong Server Core không lỗi thiếu DLL (bước kiểm tra đã đặt trong Dockerfile).
- Hạ tầng OpenStack có bật ảo hóa lồng nhau hay không (chỉ cần nếu dùng Hyper-V isolation).
- Độ trễ p95/p99 trên flavor thực tế, CPU so với GPU.
- Phiên bản Python ghim trong `Dockerfile.windows` còn bộ cài Windows trên python.org.
