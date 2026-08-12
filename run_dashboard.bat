@echo off
call "C:\Users\idos0\anaconda3\Scripts\activate.bat" aip
cd /d "%~dp0"

echo ===================================================
echo   Starting DogFight Replay Dashboard...
echo   Current Directory: %CD%
echo   URL: http://localhost:7860
echo ===================================================
echo.

python tools\dashboard.py --default-tab replay --logdir "%CD%\artifacts\logs" --port 7860

pause
