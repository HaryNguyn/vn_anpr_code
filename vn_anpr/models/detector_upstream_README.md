---
library_name: ultralytics
license: agpl-3.0
pipeline_tag: object-detection
inference: false
base_model: Ultralytics/YOLO26
base_model_relation: finetune
tags:
- yolo26
- yolo26n
- computer-vision
- license-plate-detection
- license-plate
- licence-plate-detection
- number-plate-detection
- number-plate
- alpr
- anpr
- automatic-license-plate-recognition
- automatic-number-plate-recognition
- onnx
---

# YOLO26n License Plate / Number Plate Detection

A compact, single-class YOLO26n model for detecting vehicle license plates (licence plates or number plates) in images and video frames. It returns bounding boxes and detection confidence for the plate localization stage of automatic license plate recognition (ALPR) and automatic number plate recognition (ANPR) pipelines. OCR and registration-number validation require separate models or code.

## Reference results

Evaluated at 640×640 on the dataset-provided test split containing 1,020 images and 1,085 plate instances.

| Runtime | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| PyTorch | 0.9856 | 0.9585 | 0.9642 | 0.6904 |
| ONNX Runtime CPU | 0.9830 | 0.9585 | 0.9642 | 0.6895 |

These are reference measurements on the provided dataset split, not an independent production benchmark. The ONNX export passed the PyTorch-parity check.

Reported inference time in the included evaluation files:

- PyTorch: 3.1 ms/image
- ONNX Runtime CPU: 49.5 ms/image

These values cover inference only, excluding preprocessing and postprocessing. The evaluation files do not record the full hardware configuration; these timings are not a controlled runtime comparison or a latency guarantee.

## Files

- `alpr_26n.onnx` — ONNX Runtime inference, including CPU deployment
- `alpr_26n.pt` — Ultralytics checkpoint for inference or further training
- `evaluation_metrics.json` — full PyTorch evaluation values
- `onnx_evaluation_metrics.json` — full ONNX evaluation and parity values

## Usage

Use the loading example below. Hugging Face's automatic **“Use this model”** snippet is currently incompatible with YOLO26: `YOLOvXX` and `YOLOv26.from_pretrained(...)` are not supported loading APIs. Download the weight file with `hf_hub_download`, then load it with `ultralytics.YOLO`.

```bash
pip install "ultralytics==8.4.155" "onnxruntime>=1.20" "huggingface_hub>=0.30"
```

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(
    repo_id="ishaqinu/yolo26n-license-plate-detection",
    filename="alpr_26n.onnx",
)

model = YOLO(model_path, task="detect")
results = model.predict("vehicle.jpg", imgsz=640, conf=0.25, device="cpu")

for result in results:
    print(result.boxes.xyxy.cpu().numpy())  # x1, y1, x2, y2 in image pixels
    print(result.boxes.conf.cpu().numpy())  # detection confidence
```

For PyTorch inference or further training, change `filename` to `"alpr_26n.pt"` and load the downloaded file with the same `YOLO(model_path, task="detect")` call. Replace `"vehicle.jpg"` with the path to your image or video.

Tune the confidence threshold using footage from the target cameras.

## Model details

- Class: `License_Plate`
- Input: float32 NCHW `[1, 3, 640, 640]`
- Raw ONNX output: `[1, 5, 8400]`
- ONNX batch size: fixed at 1
- Embedded NMS: no

The ONNX export returns raw box predictions and scores. The Ultralytics example handles resizing with letterboxing, confidence filtering, non-maximum suppression (NMS), and mapping boxes back to the original image.

## Limitations

Detection quality can decrease for tiny, blurred, occluded, steep-angle, reflective, overexposed, low-light, or two-line plates. The reference evaluation does not establish accuracy for every country or CCTV condition, including Indian plates. Validate on representative footage from your cameras. Detection confidence is not OCR confidence.

## Training data and license

Fine-tuned from `yolo26n.pt` using version 11 of the [License Plate Recognition dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11). The dataset page reported CC BY 4.0 when this release was prepared.

The model is released under AGPL-3.0. Review Ultralytics enterprise licensing if AGPL-3.0 is unsuitable for your deployment.
