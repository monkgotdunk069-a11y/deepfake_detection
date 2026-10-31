@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo  Starting DeepDetect AI (Backend + React Frontend)
echo ========================================================
echo.

:: 1. Launch backend in a separate window
start "DeepDetect Backend (FastAPI)" cmd /k "%~dp0run_server.bat"

echo Waiting 3 seconds for backend to start...
timeout /t 3 /nobreak >nul

:: 2. Launch frontend in a separate window
start "DeepDetect Frontend (Vite)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo Opening browser at http://127.0.0.1:5173 ...
start http://127.0.0.1:5173

