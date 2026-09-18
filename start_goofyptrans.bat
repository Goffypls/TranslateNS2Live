@echo off
setlocal
title GoofypTrans
cd /d "%~dp0"

echo ============================================
echo   GoofypTrans - levantando servidor y cliente
echo ============================================

echo.
echo [1/3] Levantando el servidor (Docker)...
docker compose up -d
if errorlevel 1 (
    echo.
    echo No se pudo levantar Docker. Fijate que Docker Desktop este abierto y reintenta.
    pause
    exit /b 1
)

echo.
echo [2/3] Esperando que el servidor termine de cargar los modelos...
:wait_health
curl -s -o nul -w "%%{http_code}" http://localhost:8000/health > "%TEMP%\goofyptrans_health.txt" 2>nul
set /p HEALTH_CODE=<"%TEMP%\goofyptrans_health.txt"
if not "%HEALTH_CODE%"=="200" (
    timeout /t 2 /nobreak >nul
    goto wait_health
)

echo.
echo [3/3] Abriendo el cliente...
call .venv\Scripts\activate.bat
python main.py

endlocal
