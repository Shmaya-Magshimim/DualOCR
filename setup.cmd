@echo off
rem DualOCR setup for Windows. Usage: setup.cmd [--help for options]
setlocal
rem Prefer PowerShell 7. Windows PowerShell 5.1 started from PowerShell 7 inherits
rem a module path it cannot load, so clear that first.
where pwsh >nul 2>nul
if %errorlevel%==0 (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
) else (
  set "PSModulePath="
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
)
exit /b %errorlevel%
