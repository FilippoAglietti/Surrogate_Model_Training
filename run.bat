@echo off
title Surrogate Model Trainer
echo.
echo   ============================================
echo    SURROGATE MODEL TRAINING ENGINE
echo    Starting up...
echo   ============================================
echo.

REM ── Check Python ──────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH.
    echo         Install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM ── Create venv if it doesn't exist ───────────
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK]   Virtual environment created.
) else (
    echo [OK]   Virtual environment found.
)

REM ── Activate venv ─────────────────────────────
call venv\Scripts\activate.bat

REM ── Install / update dependencies ─────────────
echo [INFO] Installing dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK]   Dependencies installed.

echo.
echo   ============================================
echo    Launching Streamlit app...
echo    URL: http://localhost:8501
echo   ============================================
echo.

REM ── Launch app ────────────────────────────────
streamlit run app.py

pause
