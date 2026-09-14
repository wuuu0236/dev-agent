@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY=E:\anacanda1\envs\dev-agent\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================
echo   DataLens - starting API + React frontend
echo   URL: http://127.0.0.1:8000/app
echo   Press Ctrl+C to stop
echo ============================================

start "" cmd /c "timeout /t 10 >nul & start http://127.0.0.1:8000/app"

"%PY%" -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000

pause
