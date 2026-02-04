@echo off
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=%PROJECT_ROOT%python_embeded\python.exe"

echo Checking environment...

if not exist "%PYTHON_EXE%" (
    echo [ERROR] python_embeded folder not found in:
    echo %PROJECT_ROOT%python_embeded
    echo.
    echo Please ensure you are running this from the portable package folder.
    echo If you need the portable package, please download it from the releases.
    pause
    exit /b 1
)

echo Found python_embeded. Starting ACE-Step 1.5...
echo.

"%PYTHON_EXE%" -m acestep.acestep_v15_pipeline %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    pause
)
