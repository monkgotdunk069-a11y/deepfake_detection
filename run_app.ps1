# DeepDetect AI Launcher (PowerShell)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting DeepDetect AI (Backend + React Frontend)     " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Start FastAPI backend in a separate PowerShell window
Start-Process powershell -ArgumentList "-NoExit", "-File", "$scriptDir\run_server.ps1"

Write-Host "Waiting 3 seconds for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# 2. Start Vite frontend in a separate PowerShell window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir\frontend'; npm run dev"

Write-Host "Opening browser at http://127.0.0.1:5173..." -ForegroundColor Green
Start-Process "http://127.0.0.1:5173"

