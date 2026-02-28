@echo off
echo Running nvidia-smi as Administrator...
echo.
powershell -Command "Start-Process nvidia-smi -ArgumentList '-L' -Verb RunAs -Wait"
echo.
echo Check the window that opened for GPU listing.
pause
