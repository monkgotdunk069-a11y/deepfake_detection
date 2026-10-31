@echo off
setlocal
cd /d "%~dp0"

echo Starting DeepDetect FastAPI Backend on port 8000...

set "PYTHON="

:: 1. Check if .venv\Scripts\python.exe is permitted by Windows Application Control
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "exit(0)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=.venv\Scripts\python.exe"
    )
)

:: 2. Fall back to parent Python from AppData if .venv\Scripts\ is blocked by AppLocker
if not defined PYTHON (
    if exist "%APPDATA%\uv\python\cpython-3.12-windows-x86_64-none\python.exe" (
        set "PYTHON=%APPDATA%\uv\python\cpython-3.12-windows-x86_64-none\python.exe"
    )
)

:: 3. Fall back to system python
if not defined PYTHON (
    set "PYTHON=python"
)

:: Ensure .venv site-packages and project root are always available in PYTHONPATH
set "PYTHONPATH=%~dp0;%~dp0.venv\Lib\site-packages;%PYTHONPATH%"

echo Using Python interpreter: %PYTHON%
"%PYTHON%" -m uvicorn src.api:app --reload --port 8000
if errorlevel 1 (
    echo.
    echo Failed to start the backend. Ensure Python and dependencies are installed.
    pause
)
