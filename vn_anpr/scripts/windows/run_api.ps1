<#  Chạy dịch vụ REST trên Windows.  Biến môi trường đặt trong file .env.ps1 (không đưa vào git).  #>
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..'); Set-Location $root
if (Test-Path .\.env.ps1) { . .\.env.ps1 }
if (-not $env:ANPR_CONFIG) { $env:ANPR_CONFIG = Join-Path $root 'config.yaml' }
if (-not $env:ANPR_PORT)   { $env:ANPR_PORT = '8080' }
& .\.venv\Scripts\python.exe -m uvicorn service.app:app --host 0.0.0.0 --port $env:ANPR_PORT --workers 1
