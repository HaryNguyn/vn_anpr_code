<#  Cài đặt VN-ANPR chạy trực tiếp trên Windows (không Docker) – dùng cho máy phát triển, kiểm thử hoặc máy edge tại bãi.
    Yêu cầu: Python 3.10–3.12 64-bit đã cài (python.org), PowerShell 5.1+.
    Chạy:  powershell -ExecutionPolicy Bypass -File scripts\windows\setup_native.ps1           #>
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $root
if (Get-Command py -ErrorAction SilentlyContinue) { $py = 'py -3' } else { $py = 'python' }
Write-Host "Thu muc: $root"
& cmd /c "$py -m venv .venv"
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[service,dev]"
& .\.venv\Scripts\python.exe -c "import cv2, onnxruntime; print('cv2', cv2.__version__, 'ort', onnxruntime.__version__, onnxruntime.get_available_providers())"
& .\.venv\Scripts\python.exe tests\test_core.py
& .\.venv\Scripts\python.exe tests\test_service.py
New-Item -ItemType Directory -Force -Path models, runtime | Out-Null
Write-Host "Hoan tat. Chep plate_det.onnx, plate_ocr.onnx vao thu muc models\ roi chay run_api.ps1"
