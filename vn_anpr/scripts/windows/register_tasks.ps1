<#  Đăng ký API và worker tự khởi động cùng Windows, tự chạy lại khi lỗi (Task Scheduler – có sẵn trong Windows,
    không cần phần mềm bên thứ ba). Chạy bằng quyền Administrator.
    Lưu ý: tài khoản chạy tác vụ nên là tài khoản dịch vụ riêng, không dùng Administrator.  #>
param([string]$User = 'NT AUTHORITY\SYSTEM', [switch]$Worker)
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
            -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
$trigger = New-ScheduledTaskTrigger -AtStartup
$tasks = @{ 'VN-ANPR-API' = 'run_api.ps1' }
if ($Worker) { $tasks['VN-ANPR-Worker'] = 'run_worker.ps1' }
foreach ($name in $tasks.Keys) {
  $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
            -Argument ("-NoProfile -ExecutionPolicy Bypass -File `"{0}`"" -f (Join-Path $root "scripts\windows\$($tasks[$name])")) `
            -WorkingDirectory $root
  Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -User $User -RunLevel Highest -Force | Out-Null
  Start-ScheduledTask -TaskName $name
  Write-Host "Da dang ky va khoi dong: $name"
}
