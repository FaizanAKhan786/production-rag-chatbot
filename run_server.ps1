# PowerShell launch script for RAG Chatbot Server
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting Production RAG Chatbot Server" -ForegroundColor Green
Write-Host "URL: http://localhost:8000" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

Set-Location -Path $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
