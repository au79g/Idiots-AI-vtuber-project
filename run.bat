@echo off
REM Quick start - just runs the AI (for when stage is already in OBS)
title AI VTuber
call venv\Scripts\activate.bat 2>nul
python ai_persona.py
pause
