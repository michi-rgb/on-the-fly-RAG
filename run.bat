@echo off
REM On-the-fly RAG System - Startup Script for Windows

echo.
echo ========================================
echo On-the-fly RAG System Startup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    exit /b 1
)

REM Check if Ollama is running
echo Checking Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo *** WARNING ***
    echo Ollama is not running at localhost:11434
    echo.
    echo Please start Ollama in a separate terminal:
    echo   ollama serve
    echo.
    echo Then ensure these models are downloaded:
    echo   ollama pull qwen3.5:latest
    echo   ollama pull nomic-embed-text-v2-moe
    echo.
    pause
) else (
    echo - Ollama OK
)

REM Start FastAPI server
echo.
echo Starting FastAPI server...
echo Server will be available at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
start /b python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

echo Waiting for server to start...
timeout /t 3 /nobreak >nul
start "" "http://localhost:8000"

echo.
echo Server is running. Press Ctrl+C to stop.
echo.
pause
