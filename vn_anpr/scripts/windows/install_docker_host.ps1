<#  Chuẩn bị Windows Server 2022 làm Docker host (Windows containers) theo hướng dẫn Microsoft Learn
    "Get started: Prep Windows for containers". Script của Microsoft bật tính năng Containers và KHỞI ĐỘNG LẠI máy.
    Khuyến nghị: chạy một lần khi dựng ảnh mẫu (golden image) trên OpenStack, rồi Sysprep + lưu Glance.
    Bản thương mại có hỗ trợ: Mirantis Container Runtime (https://get.mirantis.com/install.ps1).  #>
$ErrorActionPreference = 'Stop'
Invoke-WebRequest -UseBasicParsing `
  'https://raw.githubusercontent.com/microsoft/Windows-Containers/Main/helpful_tools/Install-DockerCE/install-docker-ce.ps1' `
  -OutFile "$env:TEMP\install-docker-ce.ps1"
& "$env:TEMP\install-docker-ce.ps1"
