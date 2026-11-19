<#  Build và chạy VN-ANPR dạng Windows container trên Windows Server 2022 (Docker CE/MCR).
    Ví dụ:  .\build_and_run_containers.ps1 -Registry registry.vdtc.local -Tag 1.0.0 -DataRoot D:\anpr -Worker  #>
param([string]$Registry = 'local', [string]$Tag = '1.0.0', [string]$DataRoot = 'D:\anpr',
      [int]$Port = 8080, [switch]$Push, [switch]$Worker, [switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..'); Set-Location $root
if ((docker info --format '{{.OSType}}') -ne 'windows') { throw 'Docker dang o che do Linux containers' }
$image = "$Registry/vn-anpr:$Tag-ltsc2022"
if (-not $SkipBuild) { docker build -f Dockerfile.windows -t $image . ; if ($LASTEXITCODE) { throw 'Build loi' } }
if ($Push) { docker push $image }
if (Test-Path .\.env.ps1) { . .\.env.ps1 }
New-Item -ItemType Directory -Force -Path "$DataRoot\models", "$DataRoot\runtime", "$DataRoot\conf" | Out-Null
# Windows container chỉ bind-mount thư mục -> cấu hình để trong $DataRoot\conf (chỉ chép khi chưa có)
foreach ($f in 'config.yaml', 'lanes.yaml') { if (-not (Test-Path "$DataRoot\conf\$f")) { Copy-Item ".\$f" "$DataRoot\conf\" } }

docker rm -f anpr-api 2>$null | Out-Null
docker run -d --name anpr-api --isolation process --restart unless-stopped -p "${Port}:8080" `
  -e ANPR_CONFIG=C:\app\conf\config.yaml -e "ANPR_API_KEY=$env:ANPR_API_KEY" -e "ANPR_MODEL_VERSION=$env:ANPR_MODEL_VERSION" `
  -v "$DataRoot\models:C:\app\models" -v "$DataRoot\runtime:C:\app\runtime" -v "$DataRoot\conf:C:\app\conf" $image
if ($Worker) {
  docker rm -f anpr-worker 2>$null | Out-Null
  docker run -d --name anpr-worker --isolation process --restart unless-stopped `
    -e "PARKING_WEBHOOK_URL=$env:PARKING_WEBHOOK_URL" -e "PARKING_API_KEY=$env:PARKING_API_KEY" `
    -e "CAM_USER=$env:CAM_USER" -e "CAM_PASS=$env:CAM_PASS" -e "ANPR_MODEL_VERSION=$env:ANPR_MODEL_VERSION" `
    -v "$DataRoot\models:C:\app\models" -v "$DataRoot\runtime:C:\app\runtime" `
    -v "$DataRoot\conf:C:\app\conf" $image python -m service.stream_worker --config C:\app\conf\config.yaml --lanes C:\app\conf\lanes.yaml
}
Start-Sleep 10; Invoke-RestMethod "http://127.0.0.1:$Port/health"
