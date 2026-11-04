# DeepDetect AI Docker Launcher (PowerShell)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  DeepDetect AI - Docker Launcher" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Check Docker daemon availability
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Docker Desktop is not running or not installed." -ForegroundColor Red
        Write-Host "Please start Docker Desktop and run this script again." -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "[ERROR] Docker command not found. Ensure Docker Desktop is installed." -ForegroundColor Red
    exit 1
}

Write-Host "Building and launching container via docker-compose..." -ForegroundColor Green
docker-compose up --build -d

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host "  DeepDetect AI is running inside Docker!" -ForegroundColor Green
    Write-Host "  URL: http://localhost:8000" -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host ""
    Start-Process "http://localhost:8000"
    Write-Host "Useful commands:" -ForegroundColor Cyan
    Write-Host "  - View live logs:   docker-compose logs -f"
    Write-Host "  - Stop container:   docker-compose down"
    Write-Host "  - Restart:          docker-compose restart"
} else {
    Write-Host "[ERROR] docker-compose failed to build or start." -ForegroundColor Red
}

