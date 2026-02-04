@echo off
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
set "UV_EXE=%PROJECT_ROOT%uv.exe"
set "VENV_DIR=%PROJECT_ROOT%.venv"

echo ACE-Step Local Environment Setup
echo ================================
echo.

cd /d "%PROJECT_ROOT%"

REM 1. Download UV if not present
if not exist "%UV_EXE%" (
    echo [1/3] Downloading uv...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile 'uv.zip'"
    powershell -Command "Expand-Archive -Path 'uv.zip' -DestinationPath 'uv_temp' -Force"
    move /y "uv_temp\uv-x86_64-pc-windows-msvc\uv.exe" "%PROJECT_ROOT%" >nul
    rmdir /s /q "uv_temp"
    del "uv.zip"
    if not exist "%UV_EXE%" (
        echo [ERROR] Failed to download uv.exe
        pause
        exit /b 1
    )
) else (
    echo [1/3] uv.exe already present.
)

REM 2. Create Virtual Environment
echo [2/3] Creating isolated virtual environment in .venv...
"%UV_EXE%" venv --python 3.11

if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

REM 3. Install Dependencies
echo [3/3] Installing dependencies (this may take a few minutes)...
echo.
REM We install essential dependencies for the thin client first
"%UV_EXE%" pip install gradio requests python-dotenv

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ================================================
echo Setup Complete!
echo You can now use run_colab_client.bat
echo ================================================
pause
