Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"
$venvDir = Join-Path $backendDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

function Require-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

Write-Host "Checking required tools..." -ForegroundColor Cyan
Require-Command python
Require-Command npm

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating backend virtual environment..." -ForegroundColor Cyan
    Push-Location $backendDir
    try {
        python -m venv .venv
    }
    finally {
        Pop-Location
    }
}

Write-Host "Installing backend requirements..." -ForegroundColor Cyan
Push-Location $backendDir
try {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
}
finally {
    Pop-Location
}

Write-Host "Installing frontend packages..." -ForegroundColor Cyan
Push-Location $frontendDir
try {
    npm install
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Install complete." -ForegroundColor Green
Write-Host "Next step:" -ForegroundColor Yellow
Write-Host "Double-click run.bat or run scripts\run-local.ps1" -ForegroundColor Yellow
