@echo off
rem DualOCR setup for Windows. Usage: setup.cmd [--help for options]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
exit /b %errorlevel%
