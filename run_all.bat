@echo off
cd /d "%~dp0"
python -m evaluation.run_all
if errorlevel 1 (
  echo Evaluation failed. See the error message above.
  pause
  exit /b 1
)
echo Evaluation finished. Results are in outputs\
pause
