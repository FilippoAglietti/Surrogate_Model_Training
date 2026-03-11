@echo off
echo ==============================================
echo SURROGATE BUILDER - Environment Setup Check
echo ==============================================

if not exist "venv\" (
    echo [INFO] Python virtual environment not found. Creating 'venv'...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Please ensure Python is installed and in your PATH.
        pause
        exit /b %errorlevel%
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [OK] Virtual environment 'venv' found.
)

echo.
echo [INFO] Installing/Updating dependencies from requirements.txt...
call venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b %errorlevel%
)
echo [OK] Dependencies are up to date.

echo.
echo ==============================================
echo [STARTING] Launching Surrogate Builder...
echo ==============================================
call venv\Scripts\python.exe app.py

pause
