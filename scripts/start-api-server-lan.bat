@echo off
setlocal EnableExtensions
title Qwen3.8-27B API Server - LAN mode (direct.sh --lan)

rem =====================================================================
rem  start-api-server-lan.bat - double-click Windows launcher that exposes
rem  the WSL2 vLLM API to your local network (LAN). Everything
rem  start-api-server.bat does (preflight + VRAM gate + direct vLLM boot
rem  inside WSL2 Ubuntu, console stays in THIS window, wsl --shutdown after
rem  stop) plus the three steps that make the LAN reachable:
rem
rem    1. vLLM binds 0.0.0.0 inside WSL (direct.sh --lan)
rem    2. Windows forwards the port to the WSL VM
rem       (netsh interface portproxy; needs administrator - this script
rem        re-launches itself elevated via UAC when it is not)
rem    3. Windows Firewall allows inbound TCP on the port
rem       (netsh advfirewall rule; cleaned up when the server stops)
rem
rem  Devices on the LAN then reach the API at
rem        http://<this-PC-LAN-IP>:8192/v1
rem  (the script prints the address; find the PC's LAN IP with `ipconfig`
rem  on the PC, or `Get-NetIPAddress` in PowerShell).
rem
rem  When the server stops, the portproxy + firewall rule are removed so a
rem  stale forward to an old WSL IP (which changes across reboots) never
rem  lingers; re-starting this script re-creates them.
rem
rem  NOTE: LAN mode deliberately exposes the API to every device on your
rem  network (no auth). Only run it on a trusted network.
rem
rem  Overridable environment variables (same as start-api-server.bat):
rem    MODEL_DIR   WSL path of the model folder (default below)
rem    WSL_DISTRO  WSL distribution name (default Ubuntu)
rem    SERVE_PORT  port (default 8192)
rem    VLLM_SPEC_METHOD   e.g. qwen3_5_mtp to enable MTP speculative decoding
rem    VLLM_SAMPLING_JSON JSON of server-side default sampling params
rem    VLLM_EXTRA_ARGS    extra vLLM CLI args (whitespace-separated)
rem  Command-line options are forwarded to direct.sh (e.g. --port 8001);
rem  --lan is added automatically.
rem =====================================================================

rem --- require administrator for the netsh portproxy / firewall steps ---
rem fsutil returns an error without admin rights; if missing, re-launch
rem this same script elevated (UAC prompt) and exit this non-admin copy.
fsutil dirty query %SystemDrive% >nul 2>&1
if errorlevel 1 goto need_admin
goto admin_ok

:need_admin
echo This script needs administrator rights for the Windows port forwarding
echo and firewall rule. Requesting elevation...
set "ELEV_ARGS=%*"
if not defined ELEV_ARGS goto elevate_noargs
powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs -ArgumentList '%ELEV_ARGS%'"
goto after_elevate

:elevate_noargs
powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
goto after_elevate

:after_elevate
echo.
echo The elevated copy runs in a new window. If no UAC prompt appeared, or
echo you denied it, run this script as Administrator manually.
echo Press any key to close this window.
pause
exit /b 1

:admin_ok

rem --- defaults (edit here or set the env vars above) ---
if not defined MODEL_DIR set "MODEL_DIR=/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090"
if not defined WSL_DISTRO set "WSL_DISTRO=Ubuntu"
if not defined VLLM_SAMPLING_JSON set "VLLM_SAMPLING_JSON={"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}"
if not defined FULL_MAX_MODEL_LEN set "FULL_MAX_MODEL_LEN=200000"
if not defined SERVE_PORT set "SERVE_PORT=8192"

rem --- convert this script's Windows folder to a WSL path ---
set "SCRIPT_WIN=%~dp0"
set "SCRIPT_WIN=%SCRIPT_WIN:~0,-1%"

rem --- locate the project's scripts/direct.sh (same search as start-api-server.bat) ---
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

rem --- share the optional vLLM env vars with WSL (WSLENV) ---
set "WSLENV=VLLM_SPEC_METHOD/u:VLLM_SAMPLING_JSON/u:VLLM_EXTRA_ARGS/u:FULL_MAX_MODEL_LEN/u:FULL_GPU_MEM_UTIL/u:FULL_MAX_NUM_SEQS/u:SERVE_PORT/u"

echo.
echo Starting the Qwen3.8-27B API server in LAN mode (direct.sh --lan)...
echo   launcher : %DIRECT_SH%
echo   model    : %MODEL_DIR%
echo   options  : %ARGS% --lan
echo   port     : %SERVE_PORT%
if defined VLLM_SPEC_METHOD echo   MTP      : %VLLM_SPEC_METHOD%
echo.

rem --- read the WSL VM's IP (WSL2 NAT; changes per boot) ---
for /f "delims=" %%i in ('wsl -d %WSL_DISTRO% -- hostname -I') do set "WSL_IPS=%%i"
for /f "tokens=1" %%i in ("%WSL_IPS%") do set "WSL_IP=%%i"
if not defined WSL_IP (
  echo ERROR: could not read the WSL IP (is WSL2 / distro "%WSL_DISTRO%" available?).
  pause
  exit /b 1
)
echo   WSL IP   : %WSL_IP%

rem --- read this PC's LAN IP (skip loopback, APIPA and the WSL vEthernet 172.16-31.x) ---
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$a = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^127\.' -and $_.IPAddress -notmatch '^169\.254\.' -and $_.IPAddress -notmatch '^172\.(1[6-9]|2[0-9]|3[01])\.' } | Select-Object -First 1; if ($a) { $a.IPAddress }"`) do set "LAN_IP=%%i"
if not defined LAN_IP set "LAN_IP=<this-PC-LAN-IP>"
echo   LAN IP   : %LAN_IP%  (devices use http://%LAN_IP%:%SERVE_PORT%/v1)

rem --- Windows-side forwarding: portproxy (WSL IP changes per reboot, so a
rem     stale forward from a previous run is deleted first) ---
netsh interface portproxy delete v4tov4 listenport=%SERVE_PORT% listenaddress=0.0.0.0 >nul 2>&1
netsh interface portproxy add v4tov4 listenport=%SERVE_PORT% listenaddress=0.0.0.0 connectport=%SERVE_PORT% connectaddress=%WSL_IP%
if errorlevel 1 (
  echo ERROR: could not create the portproxy rule. Check that the 'IP Helper'
  echo (iphlpsvc) service is running, then retry.
  pause
  exit /b 1
)
echo   portproxy: 0.0.0.0:%SERVE_PORT% -^> %WSL_IP%:%SERVE_PORT%

rem --- Windows Firewall: allow inbound TCP on the port ---
netsh advfirewall firewall delete rule name="Qwen3.8-27B API LAN %SERVE_PORT%" >nul 2>&1
netsh advfirewall firewall add rule name="Qwen3.8-27B API LAN %SERVE_PORT%" dir=in action=allow protocol=TCP localport=%SERVE_PORT% >nul 2>&1
if errorlevel 1 (
  echo Warning: could not add the firewall rule. LAN access may be blocked
  echo if Windows Firewall is enabled.
) else (
  echo   firewall : inbound TCP %SERVE_PORT% allowed
)

echo.
echo The server stays in this window. Press Ctrl-C to stop it; the portproxy
echo and firewall rule are removed when it stops.
echo.

rem wsl runs as a child process here: the vLLM process stays inside it, this
rem window is the service console, and after wsl exits the batch continues
rem to the cleanup below.
wsl -d %WSL_DISTRO% -- bash %DIRECT_SH% start --lan %ARGS%
set "RC=%ERRORLEVEL%"

rem --- cleanup: remove the forward + firewall rule (the WSL IP may change on
rem     the next boot; a stale forward must not linger) ---
netsh interface portproxy delete v4tov4 listenport=%SERVE_PORT% listenaddress=0.0.0.0 >nul 2>&1
netsh advfirewall firewall delete rule name="Qwen3.8-27B API LAN %SERVE_PORT%" >nul 2>&1

rem --- release the GPU VRAM held by WSL, like the other launchers ---
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
echo.
echo LAN portproxy and firewall rule removed. To serve the LAN again,
echo just double-click this script again.
echo Press any key to close this window.
pause
