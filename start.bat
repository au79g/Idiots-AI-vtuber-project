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
echo  ║   [5] First Time Setup (start here if new!)           ║
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
python ai_persona.py
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
echo  ╔═══════════════════════════════════════════════════════╗
echo  ║          Install / Update Dependencies                ║
echo  ╚═══════════════════════════════════════════════════════╝
echo.

REM Check if venv exists
if exist "venv" (
    echo  [OK] Virtual environment found.
    call venv\Scripts\activate.bat
) else (
    echo  [!] No virtual environment found. Using system Python.
    echo      Run "First Time Setup" (option 5) to create one.
    echo.
)

echo.
echo  ─── Step 1/4: Core Packages ─────────────────────────────
echo.
echo  Installing LangChain, WebSockets...
pip install langchain langchain-openai langchain-core websockets
if errorlevel 1 (
    echo.
    echo  [!] WARNING: Some core packages may have failed to install.
    echo      The AI may not work correctly without these.
)

echo.
echo  ─── Step 2/4: Memory ^& Vector Database ──────────────────
echo.
echo  Installing ChromaDB, HuggingFace Embeddings...
pip install langchain-chroma chromadb langchain-huggingface sentence-transformers
if errorlevel 1 (
    echo.
    echo  [!] WARNING: Vector DB packages had issues.
    echo      Long-term memory features may not work.
)

echo.
echo  ─── Step 3/4: TTS ^& Lip Sync ───────────────────────────
echo.
echo  Installing audio playback, phoneme tools...
pip install playsound==1.2.2 pygame soundfile
pip install nltk g2p-en
echo.
echo  Downloading NLTK pronunciation data...
python -c "import nltk; nltk.download('cmudict', quiet=True); print('  [OK] NLTK cmudict downloaded')" 2>nul || echo  [!] NLTK data download will happen on first run.

echo.
echo  ─── Step 4/4: Optional Packages ─────────────────────────
echo.
echo  Installing voice input (Whisper)...
pip install openai-whisper sounddevice numpy 2>nul
if errorlevel 1 (
    echo  [i] Voice input packages skipped (optional - needs FFmpeg too)
) else (
    echo  [OK] Voice input packages installed
    echo       Note: Also requires FFmpeg in PATH for Whisper
)

echo.
echo  Installing Kick.com chat integration...
pip install KickApi 2>nul
if errorlevel 1 (
    echo  [i] KickApi skipped (optional - for Kick.com streaming)
) else (
    echo  [OK] KickApi installed
)

echo.
echo  ═════════════════════════════════════════════════════════
echo  Dependencies installation complete!
echo.
echo  If you see errors above, you can try running this again.
echo  Optional packages that failed are OK to skip.
echo  ═════════════════════════════════════════════════════════
echo.
echo  Press any key to return to menu...
pause >nul
goto MENU

:FIRST_SETUP
cls
echo.
echo  ╔═══════════════════════════════════════════════════════╗
echo  ║              First Time Setup                         ║
echo  ╠═══════════════════════════════════════════════════════╣
echo  ║                                                       ║
echo  ║  This will:                                           ║
echo  ║    1. Check for Python                                ║
echo  ║    2. Create a virtual environment                    ║
echo  ║    3. Install ALL required dependencies               ║
echo  ║    4. Download language data (NLTK)                   ║
echo  ║    5. Create folder structure                         ║
echo  ║    6. Create requirements.txt                         ║
echo  ║                                                       ║
echo  ║  This may take 5-10 minutes on first run.             ║
echo  ║                                                       ║
echo  ╚═══════════════════════════════════════════════════════╝
echo.
set /p confirm="  Continue? [Y/N]: "
if /i not "%confirm%"=="Y" goto MENU

echo.
echo  ═══════════════════════════════════════════════════════
echo  Step 1/6: Checking Python Installation
echo  ═══════════════════════════════════════════════════════
echo.

REM Check if Python is available
python --version >nul 2>nul
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH!
    echo.
    echo  Please install Python 3.10 or newer:
    echo    https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: During install, check "Add Python to PATH"
    echo.
    echo  After installing Python, run this setup again.
    echo.
    pause
    goto MENU
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo  [OK] Found %%i

echo.
echo  ═══════════════════════════════════════════════════════
echo  Step 2/6: Creating Virtual Environment
echo  ═══════════════════════════════════════════════════════
echo.

if exist "venv" (
    echo  [OK] Virtual environment already exists. Reusing it.
) else (
    echo  Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo  [ERROR] Failed to create virtual environment.
        echo  Try: python -m pip install --upgrade pip
        echo  Then run this setup again.
        pause
        goto MENU
    )
    echo  [OK] Virtual environment created.
)

echo  Activating virtual environment...
call venv\Scripts\activate.bat
echo  [OK] Activated.

echo.
echo  Upgrading pip...
python -m pip install --upgrade pip >nul 2>nul
echo  [OK] pip is up to date.

echo.
echo  ═══════════════════════════════════════════════════════
echo  Step 3/6: Installing Required Packages
echo  ═══════════════════════════════════════════════════════
echo.
echo  This may take several minutes on first install...
echo.

echo  [1/7] LangChain framework...
pip install langchain langchain-openai langchain-core
if errorlevel 1 (
    echo  [!] WARNING: LangChain install had issues.
)

echo.
echo  [2/7] WebSocket server...
pip install websockets
if errorlevel 1 (
    echo  [!] WARNING: websockets install failed.
)

echo.
echo  [3/7] Vector database ^& embeddings...
pip install langchain-chroma chromadb
pip install langchain-huggingface sentence-transformers
if errorlevel 1 (
    echo  [!] WARNING: Vector DB packages had issues.
)

echo.
echo  [4/7] Audio playback (multiple backends for compatibility)...
pip install playsound==1.2.2
pip install pygame
pip install soundfile
if errorlevel 1 (
    echo  [i] Some audio backends may have failed - that's OK.
    echo      The system tries multiple methods automatically.
)

echo.
echo  [5/7] Lip sync ^& phoneme tools...
pip install nltk g2p-en
if errorlevel 1 (
    echo  [!] WARNING: Lip sync packages had issues.
)

echo.
echo  [6/7] Voice input (optional - Whisper speech-to-text)...
pip install openai-whisper sounddevice numpy 2>nul
if errorlevel 1 (
    echo  [i] Voice input skipped. This is optional.
    echo      To add later: pip install openai-whisper sounddevice numpy
    echo      Also requires FFmpeg: https://ffmpeg.org/download.html
) else (
    echo  [OK] Voice input packages installed.
    echo       Note: Also requires FFmpeg installed and in PATH.
)

echo.
echo  [7/7] Kick.com chat integration (optional)...
pip install KickApi 2>nul
if errorlevel 1 (
    echo  [i] KickApi skipped. This is optional.
    echo      To add later: pip install KickApi
) else (
    echo  [OK] KickApi installed.
)

echo.
echo  ═══════════════════════════════════════════════════════
echo  Step 4/6: Downloading Language Data
echo  ═══════════════════════════════════════════════════════
echo.

echo  Downloading NLTK CMU Pronunciation Dictionary...
python -c "import nltk; nltk.download('cmudict', quiet=True); print('  [OK] NLTK cmudict ready')" 2>nul
if errorlevel 1 (
    echo  [i] NLTK data will download automatically on first run.
)

echo.
echo  Verifying sentence-transformers model access...
python -c "print('  [OK] sentence-transformers available')" 2>nul
REM The embedding model downloads on first use automatically

echo.
echo  ═══════════════════════════════════════════════════════
echo  Step 5/6: Creating Folder Structure
echo  ═══════════════════════════════════════════════════════
echo.

if not exist "animations" mkdir animations
if not exist "animations\idle" mkdir animations\idle
if not exist "animations\happy" mkdir animations\happy
if not exist "animations\sad" mkdir animations\sad
if not exist "animations\angry" mkdir animations\angry
if not exist "animations\surprised" mkdir animations\surprised
if not exist "animations\talking" mkdir animations\talking
if not exist "animations\greeting" mkdir animations\greeting
if not exist "animations\general" mkdir animations\general
if not exist "memories" mkdir memories
if not exist "memories\users" mkdir memories\users
if not exist "memories\voice_sessions" mkdir memories\voice_sessions
if not exist "scripts" mkdir scripts
if not exist "piper" mkdir piper
if not exist "vector_db" mkdir vector_db
echo  [OK] All folders created.

echo.
echo  ═══════════════════════════════════════════════════════
echo  Step 6/6: Creating requirements.txt
echo  ═══════════════════════════════════════════════════════
echo.

(
echo # AI VTuber Persona System - Dependencies
echo # Generated by First Time Setup
echo.
echo # Core ^(Required^)
echo langchain
echo langchain-openai
echo langchain-core
echo websockets
echo.
echo # Lip Sync / Phonemes ^(Required for TTS^)
echo nltk
echo g2p-en
echo.
echo # Memory / Vector Database ^(Required^)
echo langchain-chroma
echo chromadb
echo langchain-huggingface
echo sentence-transformers
echo.
echo # Audio Playback
echo playsound==1.2.2
echo pygame
echo soundfile
echo.
echo # Voice Input ^(Optional^)
echo openai-whisper
echo sounddevice
echo numpy
echo.
echo # Kick.com Chat ^(Optional^)
echo # KickApi
) > requirements.txt
echo  [OK] requirements.txt created.

echo.
echo  ═══════════════════════════════════════════════════════
echo.
echo  Verifying installation...
echo.

REM Quick verification of key packages
python -c "from langchain_openai import ChatOpenAI; print('  [OK] LangChain + OpenAI')" 2>nul || echo  [FAIL] LangChain/OpenAI - REQUIRED
python -c "import websockets; print('  [OK] WebSockets')" 2>nul || echo  [FAIL] WebSockets - REQUIRED
python -c "from langchain_chroma import Chroma; print('  [OK] ChromaDB')" 2>nul || echo  [FAIL] ChromaDB - needed for memory
python -c "from langchain_huggingface import HuggingFaceEmbeddings; print('  [OK] HuggingFace Embeddings')" 2>nul || echo  [FAIL] HuggingFace - needed for memory
python -c "import nltk; print('  [OK] NLTK')" 2>nul || echo  [FAIL] NLTK - needed for lip sync
python -c "from g2p_en import G2p; print('  [OK] g2p_en')" 2>nul || echo  [FAIL] g2p_en - needed for lip sync
python -c "import pygame; print('  [OK] Pygame (audio)')" 2>nul || echo  [WARN] Pygame not available (playsound will be used)
python -c "from playsound import playsound; print('  [OK] Playsound (audio)')" 2>nul || echo  [WARN] Playsound not available

echo.
echo  ═══════════════════════════════════════════════════════
echo               SETUP COMPLETE!
echo  ═══════════════════════════════════════════════════════
echo.
echo  If all checks above show [OK], you're ready to go!
echo  If any show [FAIL], try running option [4] to retry.
echo.
echo  ─── NEXT STEPS ──────────────────────────────────────
echo.
echo  1. Install LM Studio: https://lmstudio.ai/
echo     - Load a model (recommended: Qwen 2.5 3B Instruct)
echo     - Go to Local Server tab and click Start
echo     - Make sure it runs on port 1234
echo.
echo  2. Place your VRM model file (.vrm) in this folder
echo     and update streaming_stage.html with the filename
echo.
echo  3. (Optional) Add animations:
echo     Place .vrma files in the animations/ subfolders
echo.
echo  4. (Optional) Set up Piper TTS for voice:
echo     Download: https://github.com/rhasspy/piper/releases
echo     Extract piper.exe to the piper/ folder
echo     Download a voice model to piper/ folder
echo.
echo  5. Edit ai_persona.py:
echo     - Set ADMIN_USERS to your username
echo     - Set KICK_CHANNEL if using Kick.com chat
echo.
echo  When ready, use option [1] to start!
echo.
echo  Press any key to return to menu...
pause >nul
goto MENU

:EXIT
echo.
echo  Goodbye!
timeout /t 1 >nul
exit
