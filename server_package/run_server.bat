@echo off
title Laguna S 2.1 MoE Bitplane Server (Port 8000)
echo ==================================================================
echo   Laguna S 2.1 (118B MoE) Bitplane Native C++ Local Server
echo ==================================================================
echo.
echo Installing / checking requirements...
python -m pip install -r requirements.txt
echo.
echo Starting Local Server API at http://localhost:8000 ...
python start_server.py
pause
