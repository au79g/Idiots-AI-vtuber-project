@echo off
title AI VTuber Launcher
color 0A

:MENU
cls
echo.
echo  ╔═══════════════════════════════════════════════════════╗
echo  ║           AI VTuber Persona System v5.5               ║
echo  ╠═══════════════════════════════════════════════════════╣
echo  ║                                                       ║
echo  ║   [1] Start AI + Open Stage (recommended)             ║
echo  ║   [2] Start AI Only (if stage already open in OBS)    ║
echo  ║   [3] Open Stage Only (browser)                       ║
echo  ║                                                       ║
echo  ║   [4] Install/Update Dependencies                     ║
echo  ║   [5] First Time Setup                                ║
echo  ║                                                       ║
echo  ║   [0] Exit                                            ║
echo  ║                                                       ║
echo  ╚═══════════════════════════════════════════════════════╝
echo.
set /p choice="  Enter choice [0-5]: "

if "%choice%"=="1" goto START_BOTH
if "%choice%"=="2" goto START_AI
if "%choice%"=="3" goto START_STAGE
if "%choice%"=="4" goto INSTALL_DEPS
if "%choice%"=="5" goto FIRST_SETUP
if "%choice%"=="0" goto EXIT

echo.
echo  Invalid choice. Press any key to try again...
pause >nul
goto MENU

:START_BOTH
cls
echo.
echo  Starting AI VTuber...
echo.
echo  Opening Stage in browser...
start "" "streaming_stage.html"
timeout /t 2 >nul
echo  Starting Python script...
echo.
echo  ─────────────────────────────────────────────────────────
call venv\Scripts\activate.bat 2>nul || echo  (No venv found, using system Python)
python ai_persona_v5_5.py
pause
goto MENU

:START_AI
cls
echo.
echo  Starting AI VTuber (AI only)...
echo.
echo  Note: Make sure the stage is open in OBS or browser
echo.
echo  ─────────────────────────────────────────────────────────
call venv\Scripts\activate.bat 2>nul || echo  (No venv found, using system Python)
python ai_persona.py
pause
goto MENU

:START_STAGE
cls
echo.
echo  Opening Stage in default browser...
start "" "streaming_stage.html"
echo.
echo  Stage opened! You can now add it as a Browser Source in OBS.
echo  URL: file:///%cd:\=/%/streaming_stage.html
echo.
echo  Press any key to return to menu...
pause >nul
goto MENU

:INSTALL_DEPS
cls
echo.
echo  Installing/Updating Dependencies...
echo  ─────────────────────────────────────────────────────────
echo.

REM Check if venv exists
if exist "venv" (
    echo  Virtual environment found.
    call venv\Scripts\activate.bat
) else (
    echo  No virtual environment found. Using system Python.
    echo  (Run "First Time Setup" to create a venv)
)

echo.
echo  Installing required packages...
pip install langchain langchain-openai langchain-huggingface chromadb
pip install websockets playsound pyttsx3

echo.
echo  Installing optional packages (voice input)...
pip install openai-whisper sounddevice numpy

echo.
echo  Installing optional packages (Kick chat)...
pip install KickApi

echo.
echo  ─────────────────────────────────────────────────────────
echo  Done! Press any key to return to menu...
pause >nul
goto MENU

:FIRST_SETUP
cls
echo.
echo  ╔═══════════════════════════════════════════════════════╗
echo  ║              First Time Setup                         ║
echo  ╚═══════════════════════════════════════════════════════╝
echo.
echo  This will:
echo    1. Create a Python virtual environment
echo    2. Install all required dependencies
echo    3. Create default config files
echo.
set /p confirm="  Continue? [Y/N]: "
if /i not "%confirm%"=="Y" goto MENU

echo.
echo  ─────────────────────────────────────────────────────────
echo  Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo.
    echo  ERROR: Failed to create venv. Is Python installed?
    echo  Download Python from: https://www.python.org/downloads/
    pause
    goto MENU
)

echo  Activating venv...
call venv\Scripts\activate.bat

echo.
echo  ─────────────────────────────────────────────────────────
echo  Step 2: Installing required packages...
pip install --upgrade pip
pip install langchain langchain-openai langchain-huggingface chromadb
pip install websockets playsound pyttsx3

echo.
echo  Step 3: Installing optional packages...
echo  (These enable voice input and Kick chat - can fail safely)
pip install openai-whisper sounddevice numpy 2>nul
pip install KickApi 2>nul

echo.
echo  ─────────────────────────────────────────────────────────
echo  Step 4: Creating folder structure...
if not exist "animations" mkdir animations
if not exist "animations\idle" mkdir animations\idle
if not exist "animations\happy" mkdir animations\happy
if not exist "animations\sad" mkdir animations\sad
if not exist "animations\angry" mkdir animations\angry
if not exist "animations\surprised" mkdir animations\surprised
if not exist "animations\talking" mkdir animations\talking
if not exist "animations\greeting" mkdir animations\greeting
if not exist "memories" mkdir memories
if not exist "memories\users" mkdir memories\users
if not exist "memories\voice_sessions" mkdir memories\voice_sessions
if not exist "scripts" mkdir scripts
if not exist "piper" mkdir piper

echo.
echo  ─────────────────────────────────────────────────────────
echo  Setup Complete!
echo.
echo  NEXT STEPS:
echo.
echo  1. Download LM Studio: https://lmstudio.ai/
echo     - Load a model (recommended: Qwen 2.5 3B Instruct)
echo     - Start local server on port 1234
echo.
echo  2. Add your VRM model:
echo     - Place .vrm file in project folder
echo     - Update streaming_stage.html with model path
echo.
echo  3. Add animations (optional):
echo     - Place .vrma files in animations/ subfolders
echo.
echo  4. Add Piper TTS (optional):
echo     - Download from: https://github.com/rhasspy/piper/releases
echo     - Extract piper.exe to piper/ folder
echo     - Download a voice model (.onnx) to piper/ folder
echo.
echo  5. Edit ai_persona_v5_5.py:
echo     - Set ADMIN_USERS to your username
echo     - Set KICK_CHANNEL to your Kick channel (if using)
echo.
echo  Press any key to return to menu...
pause >nul
goto MENU

:EXIT
echo.
echo  Goodbye!
timeout /t 1 >nul
exit
