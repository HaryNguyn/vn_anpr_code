<#  Chạy worker đọc luồng RTSP theo làn và đẩy sự kiện webhook sang phần mềm bãi đỗ.  #>
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..'); Set-Location $root
if (Test-Path .\.env.ps1) { . .\.env.ps1 }
& .\.venv\Scripts\python.exe -m service.stream_worker --config config.yaml --lanes lanes.yaml
