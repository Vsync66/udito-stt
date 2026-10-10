@echo off
REM udito.bat, one-command CLI: daemon-first with automatic one-shot fallback.
REM usage: udito.bat <input_audio> <output_txt> [language]
set "HERE=%~dp0"
where python >nul 2>nul
if %errorlevel%==0 (
    python "%HERE%main.py" transcribe %1 %2 %3
) else (
    py "%HERE%main.py" transcribe %1 %2 %3
)