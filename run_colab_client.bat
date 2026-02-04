@echo off
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"

echo Checking environment...

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment not found. 
    echo Please run 'setup_local_env.bat' first to install dependencies.
    pause
    exit /b 1
)

echo Found local environment. Starting Colab Thin Client...
echo.

"%PYTHON_EXE%" scripts\colab_thin_client.py %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    pause
)
