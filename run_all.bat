@echo off
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
  echo Project virtual environment not found: .venv
  echo Follow README sections 1-6 before running this file.
  pause
  exit /b 1
)
"%PROJECT_PYTHON%" run_all.py %*
if errorlevel 1 (
  echo Evaluation failed. See the error message above.
  pause
  exit /b 1
)
echo Pipeline finished. See outputs\ or the selected --output-dir.
pause
