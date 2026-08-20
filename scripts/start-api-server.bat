@echo off
setlocal EnableExtensions
title Qwen3.8-27B API Server (direct.sh)

rem =====================================================================
rem  start-api-server.bat - double-click Windows launcher for
rem  scripts/direct.sh: runs the same preflight + VRAM gate + direct vLLM
rem  boot inside WSL2 Ubuntu and keeps the service in THIS console window
rem  (Ctrl-C stops it). After the server stops it runs "wsl --shutdown" to
rem  fully shut down the WSL VM and release its GPU VRAM. Accepts the same
rem  options as direct.sh; without
rem  --model-dir it uses the default model folder below.
rem
rem  Overridable environment variables:
rem    MODEL_DIR   WSL path of the model folder (default below)
rem    WSL_DISTRO  WSL distribution name (default Ubuntu)
rem =====================================================================

rem --- defaults (edit here or set the env vars above) ---
if not defined MODEL_DIR  set "MODEL_DIR=/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090"
if not defined WSL_DISTRO set "WSL_DISTRO=Ubuntu"

rem --- convert this script's Windows folder to a WSL path ---
set "SCRIPT_WIN=%~dp0"
set "SCRIPT_WIN=%SCRIPT_WIN:~0,-1%"

rem --- locate the project's scripts/direct.sh ---
rem The launcher may sit anywhere (e.g. a copy on the Desktop). Try, in
rem order: an explicit PROJECT_DIR, this script's own folder upwards, then
rem the default project location on this machine.
if not defined PROJECT_DIR set "PROJECT_DIR=D:\Code\MJ-Project\ai-model-nvfp4"
set "PROJ=%PROJECT_DIR%"
if exist "%PROJ%\scripts\direct.sh" goto found_direct
set "PROJ=%SCRIPT_WIN%"
:find_direct
if exist "%PROJ%\scripts\direct.sh" goto found_direct
set "PARENT=%PROJ%"
for %%I in ("%PROJ%\.") do set "PROJ=%%~dpI"
set "PROJ=%PROJ:~0,-1%"
if "%PROJ%"=="%PARENT%" goto no_direct
goto find_direct
:no_direct
echo ERROR: scripts\direct.sh not found.
echo Looked at: %PROJECT_DIR% and every folder above %SCRIPT_WIN%.
echo Set PROJECT_DIR to the project folder if it lives elsewhere, e.g.:
echo   set PROJECT_DIR=D:\Code\MJ-Project\ai-model-nvfp4
echo.
echo Press any key to close this window.
pause
exit /b 1
:found_direct

rem Drive letter comes from the PROJECT path (the launcher may sit on a
rem different drive than the project, e.g. a Desktop copy on C:). WSL
rem mounts as lowercase (/mnt/d), so the letter is lowercased.
set "DRIVE=%PROJ:~0,1%"
for %%D in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
  if /I "%DRIVE%"=="%%D" set "DRIVE=%%D"
)
set "DIRECT_WIN=%PROJ%\scripts\direct.sh"
set "DIRECT_SH=%DIRECT_WIN:\=/%"
set "DIRECT_SH=/mnt/%DRIVE%/%DIRECT_SH:~3%"
rem wsl.exe does not strip quotes from -d, and joins everything after --
rem into one command line parsed by bash, so the path must not be quoted;
rem spaces in it are escaped for bash instead.
set "DIRECT_SH=%DIRECT_SH: =\ %"

rem --- forward command-line options; default --model-dir when absent ---
set "ARGS=%*"
echo %ARGS% | findstr /C:"--model-dir" >nul || set "ARGS=%ARGS% --model-dir %MODEL_DIR%"

echo Starting the Qwen3.8-27B API server via WSL (%WSL_DISTRO%)...
echo   launcher: %DIRECT_SH%
echo   model   : %MODEL_DIR%
echo   options : %ARGS%
echo.
echo The server stays in this window. Press Ctrl-C to stop it; the
echo window then stays open so you can read the result.
echo.

rem wsl runs as a child process here: the vLLM process stays inside it,
rem this window is the service console, and after wsl exits the batch
rem continues to the pause below (which keeps the window open).
wsl -d %WSL_DISTRO% -- bash %DIRECT_SH% start %ARGS%
set "RC=%ERRORLEVEL%"

rem The API server has stopped (Ctrl-C or otherwise). Fully shut down the
rem WSL VM so the GPU VRAM it held is released back to Windows - the same
rem "wsl --shutdown" step the troubleshooting guide recommends.
echo.
echo Stopping WSL to release VRAM...
wsl --shutdown
set "WSLRC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo API server stopped, exit code 0.
) else (
  echo API server exited with error code %RC%.
)
if "%WSLRC%"=="0" (
  echo WSL fully shut down - VRAM released.
) else (
  echo Warning: wsl --shutdown returned error code %WSLRC%.
)
echo Press any key to close this window.
pause
