@echo off
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=%PROJECT_ROOT%python_embeded\python.exe"

echo Checking environment...

if not exist "%PYTHON_EXE%" (
    echo [ERROR] python_embeded folder not found in:
    echo %PROJECT_ROOT%python_embeded
    pause
    exit /b 1
)

echo Found python_embeded. Starting Colab Thin Client...
echo.

"%PYTHON_EXE%" scripts\colab_thin_client.py %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    pause
)
