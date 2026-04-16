Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"
$venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Backend virtual environment not found. Run install.bat first."
}

if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    throw "Frontend packages not found. Run install.bat first."
}

$backendCommand = "cd /d `"$backendDir`"; & `"$venvPython`" app.py"
$frontendCommand = "cd /d `"$frontendDir`"; npm run dev"

Write-Host "Starting backend..." -ForegroundColor Cyan
Start-Process powershell.exe -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand

Write-Host "Starting frontend..." -ForegroundColor Cyan
Start-Process powershell.exe -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $frontendCommand

Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "App windows started." -ForegroundColor Green
Write-Host "If the browser does not open, visit http://localhost:5173 manually." -ForegroundColor Yellow
