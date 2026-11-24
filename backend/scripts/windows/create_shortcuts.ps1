# Put two StorePulse demo buttons on the desktop (run once):
#   powershell -ExecutionPolicy Bypass -File backend\scripts\windows\create_shortcuts.ps1
$launcher = Join-Path $PSScriptRoot "StorePulse-Demo.ps1"
$backend = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell

$buttons = @(
    @{ Name = "StorePulse Demo"; Args = ""; Note = "Start StorePulse and open it in the browser" },
    @{ Name = "StorePulse Demo (phones on Wi-Fi)"; Args = " -Phones"; Note = "Start StorePulse with a QR code for phones on the same Wi-Fi" }
)
foreach ($b in $buttons) {
    $lnk = $shell.CreateShortcut((Join-Path $desktop "$($b.Name).lnk"))
    $lnk.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $lnk.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$launcher`"$($b.Args)"
    $lnk.WorkingDirectory = $backend
    $lnk.Description = $b.Note
    $lnk.IconLocation = "$env:SystemRoot\System32\imageres.dll,186"
    $lnk.Save()
    Write-Host "created $(Join-Path $desktop "$($b.Name).lnk")"
}
