# DualOCR setup for Windows: installs everything it needs and builds the app.
# Run it with setup.cmd (which allows this script to run), or:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $uvHome = Join-Path $env:USERPROFILE ".local\bin"
    if (-not (Test-Path (Join-Path $uvHome "uv.exe"))) {
        Write-Host "==> Installing uv, which sets up Python for DualOCR (https://docs.astral.sh/uv/)"
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    $env:Path = "$uvHome;$env:Path"
}

Write-Host "==> Installing Python and DualOCR's Python packages"
uv sync --locked --extra build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv run --no-sync python scripts/setup.py @args
exit $LASTEXITCODE
