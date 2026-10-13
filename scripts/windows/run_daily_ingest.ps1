<#
.SYNOPSIS
    Run the daily ingest and the quality checks, appending all output to a log file.

.DESCRIPTION
    Changes to the repository folder, runs `uv run qr ingest` and then `uv run qr validate`,
    and appends everything both commands print to logs\ingest.log (gitignored).

    Meant to be started by Windows Task Scheduler (see register_task.ps1), but it can be
    run by hand too:

        powershell -ExecutionPolicy Bypass -File scripts\windows\run_daily_ingest.ps1

    The exit code is 0 only if both commands succeeded.

.PARAMETER LogFile
    Where to append the output. Default: logs\ingest.log in the repository.
#>
[CmdletBinding()]
param(
    [string]$LogFile
)

$ErrorActionPreference = 'Stop'

# This script lives in <repo>\scripts\windows, so the repository is two folders up.
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repo

if (-not $LogFile) {
    $LogFile = Join-Path $repo 'logs\ingest.log'
}
$logDir = Split-Path -Parent $LogFile
if ($logDir) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

function Write-Log([string]$Text) {
    # UTF-8 without a byte-order mark, the same encoding the qr output below uses.
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::AppendAllText($LogFile, "$Text`r`n", $utf8)
}

function Invoke-Logged([string]$Command) {
    Write-Log "--- $Command"
    # cmd.exe does the redirection: Windows PowerShell 5.1 would wrap every line a program
    # writes to stderr (where qr logs) in an error record.
    cmd.exe /d /c "$Command >> `"$LogFile`" 2>&1"
    $code = $LASTEXITCODE
    Write-Log "--- exit code $code"
    return $code
}

# uv is either on PATH or installed as a module of the Python that `py` starts.
$uv = $null
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $uv = 'uv'
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    cmd.exe /d /c 'py -m uv --version >nul 2>&1'
    if ($LASTEXITCODE -eq 0) {
        $uv = 'py -m uv'
    }
}

Write-Log ''
Write-Log "==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') daily ingest in $repo"

if (-not $uv) {
    Write-Log 'uv was not found. Install it (https://docs.astral.sh/uv/) and try again.'
    exit 1
}

# Plain, wide, UTF-8 output: nobody is watching a terminal.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:NO_COLOR = '1'
$env:COLUMNS = '160'

$ingest = Invoke-Logged "$uv run qr ingest"
# Validate even if the ingest had problems: the stored data is still worth checking.
$validate = Invoke-Logged "$uv run qr validate"

$worst = [Math]::Max([int]$ingest, [int]$validate)
Write-Log "==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') finished (ingest $ingest, validate $validate)"
exit $worst
