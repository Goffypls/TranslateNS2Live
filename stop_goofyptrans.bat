@echo off
setlocal
title GoofypTrans - apagar
cd /d "%~dp0"

echo Apagando el servidor de GoofypTrans (Docker)...
docker compose down

echo.
echo Listo, consumo en cero. Cerra esta ventana cuando quieras.
pause
endlocal
