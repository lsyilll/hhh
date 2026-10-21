@echo off
setlocal
"%~dp0.venv-platform\Scripts\python.exe" -u "%~dp0scripts\extended_stage.py" --stage analytics_verify
if errorlevel 1 (pause & exit /b 1)
pause
