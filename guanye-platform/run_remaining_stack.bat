@echo off
setlocal
"%~dp0.venv-platform\Scripts\python.exe" -u "%~dp0scripts\extended_stage.py" %*
if errorlevel 1 (pause & exit /b 1)
pause
