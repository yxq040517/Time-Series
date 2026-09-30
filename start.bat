@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is required. Install Python and enable Add to PATH.
  pause
  exit /b 1
)
python scripts\start.py %*
if errorlevel 1 pause
