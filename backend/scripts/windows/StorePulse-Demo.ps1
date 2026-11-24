# One-click StorePulse demo for presenting (used by the desktop shortcuts).
#   StorePulse-Demo.ps1            -> app on this laptop, browser opens when ready
#   StorePulse-Demo.ps1 -Phones    -> same, plus a QR code so phones on the same Wi-Fi can open it
#   StorePulse-Demo.ps1 -Public    -> same, plus a free public https link + QR code (any network, any phone)
# Closing: press Ctrl+C in this window (stops the app and the database).
param([switch]$Phones, [switch]$Public)

$ErrorActionPreference = "Stop"
$backend = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$root = Split-Path -Parent $backend
$python = Join-Path $backend ".venv\Scripts\python.exe"
$url = "http://localhost:3000"
$Host.UI.RawUI.WindowTitle = "StorePulse Demo - press Ctrl+C here to stop"

function Test-App {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:3000/api/v1/health/ready"
        return $r.StatusCode -eq 200
    } catch { return $false }
}

if (-not (Test-Path $python)) {
    Write-Host "StorePulse is not set up on this machine yet (missing $python). See docs\LOCAL_DEV.md." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

if (Test-App) {
    Write-Host "StorePulse is already running - opening $url"
    Start-Process $url
    if ($Phones -or $Public) {
        Write-Host "For phones or a public link, restart it with this button after stopping the running app (Ctrl+C in its window)."
    }
    Start-Sleep -Seconds 4
    exit 0
}

Write-Host "Starting StorePulse (database, API, worker, UI) ... the browser opens when it is ready." -ForegroundColor Green
Write-Host "Log in with an account from $root\.local\logins.txt (the browser remembers it afterwards)."
Write-Host ""

# Open the browser as soon as the UI answers (first start may build the UI, ~1 min).
Start-Job -ArgumentList $url -ScriptBlock {
    param($url)
    for ($i = 0; $i -lt 150; $i++) {
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:3000/api/v1/health/ready"
            if ($r.StatusCode -eq 200) { Start-Process $url; return }
        } catch { }
        Start-Sleep -Seconds 2
    }
} | Out-Null

Set-Location $backend
$mode = if ($Public) { "--public" } elseif ($Phones) { "--lan" } else { "--present" }
& $python -m scripts.local_stack up $mode
