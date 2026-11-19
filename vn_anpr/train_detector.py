#!/usr/bin/env python3
"""Huấn luyện mô hình phát hiện biển số (hộp bao, hoặc pose 4 góc) và export ONNX.
LƯU Ý BẢN QUYỀN: thư viện ultralytics theo AGPL-3.0; nhúng vào sản phẩm thương mại đóng mã cần Enterprise License
của Ultralytics, hoặc thay bằng detector giấy phép Apache/MIT và chỉ giữ chuẩn đầu ra ONNX (xem README).

Định dạng dữ liệu YOLO: data.yaml trỏ tới images/{train,val,test} và labels/*.txt
  - hộp bao: "cls cx cy w h" (chuẩn hóa 0..1)
  - pose 4 góc: "cls cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4" (TL,TR,BR,BL) và trong data.yaml: kpt_shape: [4, 3]
Ví dụ:
  python train_detector.py --data data/plate.yaml --model yolo11s-pose.pt --epochs 150 --imgsz 640
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11s.pt", help="yolo11s.pt (hộp bao) hoặc yolo11s-pose.pt (4 góc)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                # tăng cường mô phỏng góc lệch <= 30 độ, ánh sáng đủ nhưng thay đổi
                degrees=10, perspective=0.0008, shear=5, hsv_v=0.4, mosaic=1.0, fliplr=0.0,
                patience=30, cos_lr=True)
    metrics = model.val(data=a.data, split="test")
    print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)
    path = model.export(format="onnx", imgsz=a.imgsz, opset=17, simplify=True, dynamic=False)
    print("Đã export:", path, "-> sao chép thành models/plate_det.onnx và khai báo num_keypoints trong config.yaml")


if __name__ == "__main__":
    main()
