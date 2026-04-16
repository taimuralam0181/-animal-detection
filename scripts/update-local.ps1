Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"
$venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"

function Require-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

Write-Host "Checking required tools..." -ForegroundColor Cyan
Require-Command git
Require-Command python
Require-Command npm

Write-Host "Pulling latest code from GitHub..." -ForegroundColor Cyan
Push-Location $projectRoot
try {
    git pull --ff-only origin main
}
finally {
    Pop-Location
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment missing. Running install flow first..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "install-local.ps1")
    exit 0
}

Write-Host "Updating backend requirements..." -ForegroundColor Cyan
Push-Location $backendDir
try {
    & $venvPython -m pip install -r requirements.txt
}
finally {
    Pop-Location
}

Write-Host "Updating frontend packages..." -ForegroundColor Cyan
Push-Location $frontendDir
try {
    npm install
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Update complete." -ForegroundColor Green
Write-Host "Now run run.bat to start the app." -ForegroundColor Yellow
