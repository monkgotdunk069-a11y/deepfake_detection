# DeepDetect AI FastAPI Backend Launcher (PowerShell)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Starting DeepDetect FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Cyan

$pythonExe = $null

# 1. Try .venv python
if (Test-Path "$scriptDir\.venv\Scripts\python.exe") {
    try {
        & "$scriptDir\.venv\Scripts\python.exe" -c "exit(0)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonExe = "$scriptDir\.venv\Scripts\python.exe"
        }
    } catch {}
}

# 2. Try AppData uv python if AppLocker blocks OneDrive .venv
if (-not $pythonExe) {
    $uvPy = "$env:APPDATA\uv\python\cpython-3.12-windows-x86_64-none\python.exe"
    if (Test-Path $uvPy) {
        $pythonExe = $uvPy
    }
}

# 3. Fallback to system python
if (-not $pythonExe) {
    $pythonExe = "python"
}

$env:PYTHONPATH = "$scriptDir;$scriptDir\.venv\Lib\site-packages;$env:PYTHONPATH"

Write-Host "Using Python interpreter: $pythonExe" -ForegroundColor Green
& $pythonExe -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload

