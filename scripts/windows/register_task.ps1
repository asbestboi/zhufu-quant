<#
.SYNOPSIS
    Register (or remove) a Windows Task Scheduler task that runs the daily ingest.

.DESCRIPTION
    Creates a task that runs run_daily_ingest.ps1 on weekdays at a time of your choice.
    It runs as you, only while you are logged on, so it needs no administrator rights and
    no stored password. If the PC was off or asleep at the scheduled time, the task runs
    as soon as it can afterwards. Running this script again replaces the task, which is
    how you change the time.

        powershell -ExecutionPolicy Bypass -File scripts\windows\register_task.ps1
        powershell -ExecutionPolicy Bypass -File scripts\windows\register_task.ps1 -Time 22:45

    To remove the task again:

        powershell -ExecutionPolicy Bypass -File scripts\windows\register_task.ps1 -Remove

    (or: Unregister-ScheduledTask -TaskName qr-daily-ingest -Confirm:$false)

    Pick a time after the US market close (16:00 New York), because only completed
    sessions are stored. The default of 23:30 suits European time zones. The task fires on
    your local Monday to Friday, so if you live far east of New York and want a morning
    run, schedule it yourself for Tuesday to Saturday instead.

.PARAMETER Time
    Local time of day as HH:mm (24-hour clock). Default: 23:30.

.PARAMETER TaskName
    Name of the scheduled task. Default: qr-daily-ingest.

.PARAMETER Remove
    Remove the task instead of creating it.
#>
[CmdletBinding()]
param(
    [string]$Time = '23:30',
    [string]$TaskName = 'qr-daily-ingest',
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

if ($Remove) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed the scheduled task '$TaskName'."
    }
    else {
        Write-Host "There is no scheduled task called '$TaskName'."
    }
    return
}

$at = [datetime]::MinValue
$formats = [string[]]@('H:mm', 'HH:mm')
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$none = [System.Globalization.DateTimeStyles]::None
if (-not [datetime]::TryParseExact($Time, $formats, $invariant, $none, [ref]$at)) {
    throw "Time must look like 23:30 (24-hour HH:mm), got '$Time'."
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$script = Join-Path $PSScriptRoot 'run_daily_ingest.ps1'

$arguments = "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $repo
$weekdays = 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $at
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description 'zhufu-quant: download daily market data and run the quality checks.' -Force | Out-Null

Write-Host "Scheduled '$TaskName' for weekdays at $($at.ToString('HH:mm')) local time."
Write-Host "  Runs:      $script"
Write-Host "  Log file:  $(Join-Path $repo 'logs\ingest.log')"
Write-Host "  Try it:    Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Remove it: powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Remove"
