@echo off
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"

echo Checking environment...

if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=%PROJECT_ROOT%python_embeded\python.exe"
    if not exist "!PYTHON_EXE!" (
        echo [ERROR] No valid Python environment found.
        echo Please run 'setup_local_env.bat' or ensure 'python_embeded' exists.
        pause
        exit /b 1
    )
    echo Using python_embeded environment.
) else (
    echo Using local virtual environment.
)

echo Starting API Server...
echo.

REM Enable API mode and auto-init service
"%PYTHON_EXE%" -m acestep.acestep_v15_pipeline --enable-api --init_service true %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    pause
)
