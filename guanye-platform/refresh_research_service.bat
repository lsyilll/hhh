@echo off
setlocal
"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\refresh_research_service.ps1"
if errorlevel 1 (pause & exit /b 1)
pause
