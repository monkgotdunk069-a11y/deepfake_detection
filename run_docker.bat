@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   DeepDetect AI - Docker Launcher
echo ========================================================
echo.

:: Check if Docker is installed and running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running or not installed.
    echo Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)

echo Building and starting DeepDetect AI container with Docker Compose...
docker-compose up --build -d

if errorlevel 1 (
    echo.
    echo [ERROR] Docker Compose build or startup failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo  DeepDetect AI is now running inside Docker!
echo  URL: http://localhost:8000
echo ========================================================
echo.
echo Opening browser...
start http://localhost:8000

echo.
echo Useful Docker commands:
echo   - View container logs:    docker-compose logs -f
echo   - Stop container:         docker-compose down
echo   - Restart container:      docker-compose restart
echo.
pause

