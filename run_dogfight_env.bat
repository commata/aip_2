@echo off
call "C:\Users\idos0\anaconda3\Scripts\activate.bat" aip
cd /d "%~dp0"

echo =======================================================================
echo   DogFight AIP Environment Ready
echo   Current Directory: %CD%
echo =======================================================================
echo.
echo [Example Commands]
echo.
echo python run_local_dogfight.py --ownship-backend bt --ownship-bt-dll AIP_DCS_new.dll --bt-rule-xml Rule_sei_AIP2_default.xml --target-backend fixed --save-log
echo.
echo =======================================================================
echo.

cmd /k
