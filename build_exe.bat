@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0build_exe.ps1"
