@echo off
title RAG Chatbot Server
echo ============================================================
echo Starting Production RAG Chatbot Server
echo URL: http://localhost:8000
echo ============================================================

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
pause
