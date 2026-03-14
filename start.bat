echo.
@echo off
echo Starting Smart ASL Recognition...
echo.
set PYTHONNOUSERSITE=1
call "%~dp0mp_env\Scripts\activate.bat"
cd /d "%~dp0"
"%~dp0mp_env\Scripts\python.exe" app.py
pause
