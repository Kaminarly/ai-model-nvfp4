@echo off
chcp 65001 >nul 2>&1
setlocal EnableExtensions
title Qwen3.8-27B API Server (direct.sh + MTP, 180k)

rem =====================================================================
rem  start-api-server-mtp.bat - double-click Windows launcher for
rem  scripts/direct.sh with MTP enabled and the context set to 180k.
rem  Everything else is identical to start-api-server.bat: it runs the
rem  same preflight + VRAM gate + direct vLLM boot inside WSL2 Ubuntu and
rem  keeps the service in THIS console window (Ctrl-C stops it). After the
rem  server stops it runs "wsl --shutdown" to fully shut down the WSL VM
rem  and release its GPU VRAM. Accepts the same options as direct.sh;
rem  without --model-dir it uses the default model folder below.
rem
rem  Differences from start-api-server.bat:
rem    - MTP on:            VLLM_SPEC_METHOD=mtp
rem    - default sampling:  VLLM_SAMPLING_JSON={temperature:1.0, top_p:0.95,
rem                         top_k:20, min_p:0.0, presence_penalty:0.0,
rem                         repetition_penalty:1.0}
rem    - context:           180000 default (FULL_MAX_MODEL_LEN overrides)
rem    - spec tokens 3:     --spec-tokens 3 (set in scripts/lib/serve-lib.sh)
rem    - port 8192:         same as start-api-server.bat
rem
rem  Before the service starts, this launcher asks whether to enable LAN
rem  access: 1 enable / 2 disable (default) / 0 quit. Enable does the same
rem  three steps as start-api-server-lan.bat (UAC elevation if needed):
rem  vLLM --lan (bind 0.0.0.0), Windows portproxy, firewall inbound rule.
rem  An internal --lan-enabled argument skips the menu after elevation.
rem
rem  Overridable environment variables:
rem    MODEL_DIR   WSL path of the model folder (default below)
rem    WSL_DISTRO  WSL distribution name (default Ubuntu)
rem    VLLM_SPEC_METHOD   MTP / speculative method (default mtp)
rem    VLLM_SAMPLING_JSON server-side default sampling params JSON
rem    VLLM_EXTRA_ARGS    extra vLLM CLI args (whitespace-separated)
rem    FULL_MAX_MODEL_LEN context length (default 180000)
rem    FULL_GPU_MEM_UTIL  VRAM utilization (direct.sh default 0.90)
rem    FULL_MAX_NUM_SEQS  max concurrent sequences (direct.sh default 16)
rem    SERVE_PORT         port (default 8192, same as start-api-server.bat)
rem =====================================================================

rem --- defaults (edit here or set the env vars above) ---
rem This launcher always serves the default model below (no model picker);
rem override it with MODEL_DIR or --model-dir for a different checkpoint.
if not defined MODEL_DIR set "MODEL_DIR=/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090"
if not defined WSL_DISTRO set "WSL_DISTRO=Ubuntu"
if not defined VLLM_SPEC_METHOD set "VLLM_SPEC_METHOD=mtp"
if not defined VLLM_SAMPLING_JSON set "VLLM_SAMPLING_JSON={"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}"
if not defined FULL_MAX_MODEL_LEN set "FULL_MAX_MODEL_LEN=180000"
if not defined SERVE_PORT set "SERVE_PORT=8192"

rem --- LAN menu (before the service starts) ---
rem --lan-enabled is internal: the elevated copy skips the menu.
if /I "%~1"=="--lan-enabled" (
  set "ENABLE_LAN=1"
  shift
  goto after_lan_menu
)

:lan_menu
echo.
echo LAN access
echo   1 Enable
echo   2 Disable
echo   0 Quit
echo.
set "LAN_CHOICE=2"
set /p "LAN_CHOICE=Select [1/2/0] (default 2): "
if "%LAN_CHOICE%"=="" set "LAN_CHOICE=2"
if "%LAN_CHOICE%"=="1" goto lan_chosen
if "%LAN_CHOICE%"=="2" goto after_lan_menu
if "%LAN_CHOICE%"=="0" goto lan_quit
echo Invalid choice, enter 1, 2 or 0.
goto lan_menu

:lan_quit
echo Exited.
echo Press any key to close this window.
pause
exit /b 0

:lan_chosen
set "ENABLE_LAN=1"
rem netsh portproxy / firewall need administrator. Re-launch elevated
rem via UAC when this copy is not admin, then exit the non-admin copy.
fsutil dirty query %SystemDrive% >nul 2>&1
if errorlevel 1 goto need_admin
goto admin_ok

:need_admin
echo LAN access needs administrator rights (port forward + firewall). Requesting elevation...
set "ELEV_ARGS=%*"
if not defined ELEV_ARGS goto elevate_noargs
powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs -ArgumentList '--lan-enabled %ELEV_ARGS%'"
goto after_elevate

:elevate_noargs
powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs -ArgumentList '--lan-enabled'"
goto after_elevate

:after_elevate
echo.
echo The elevated copy continues in a new window.
echo If no UAC prompt appears or you denied it, right-click this script and run as Administrator.
echo Press any key to close this window.
pause
exit /b 1

:admin_ok

:after_lan_menu

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
rem Drop the internal --lan-enabled re-launch flag and any CLI --lan (the
rem menu decides; LAN_FLAG is added separately below). Do this argument by
rem argument: on an EMPTY %* the string-substitute form "set ARGS=%ARGS:--lan=%"
rem outputs a literal "--lan=" (cmd quirk), and --lan-enabled contains the
rem substring --lan, so plain substitution is unsafe either way.
set "ARGS="
:arg_loop
if "%~1"=="" goto args_done
if /I "%~1"=="--lan-enabled" goto arg_skip
if /I "%~1"=="--lan" goto arg_skip
set "ARGS=%ARGS% %~1"
:arg_skip
shift
goto arg_loop
:args_done
echo %ARGS% | findstr /C:"--model-dir" >nul || set "ARGS=%ARGS% --model-dir %MODEL_DIR%"

set "LAN_FLAG="
if defined ENABLE_LAN set "LAN_FLAG=--lan"

rem --- share the optional vLLM env vars with WSL (WSLENV) ---
set "WSLENV=VLLM_SPEC_METHOD/u:VLLM_SAMPLING_JSON/u:VLLM_EXTRA_ARGS/u:FULL_MAX_MODEL_LEN/u:FULL_GPU_MEM_UTIL/u:FULL_MAX_NUM_SEQS/u:SERVE_PORT/u"

echo Starting the Qwen3.8-27B API server via WSL (%WSL_DISTRO%)...
echo   launcher: %DIRECT_SH%
echo   model   : %MODEL_DIR%
echo   options : %LAN_FLAG% %ARGS%
if defined VLLM_SPEC_METHOD echo   MTP     : %VLLM_SPEC_METHOD%
if defined VLLM_SAMPLING_JSON echo   sampling: %VLLM_SAMPLING_JSON%
if defined FULL_MAX_MODEL_LEN echo   context : %FULL_MAX_MODEL_LEN%
if defined SERVE_PORT echo   port    : %SERVE_PORT%
if defined ENABLE_LAN goto print_lan_on
echo   LAN     : off (loopback only)
goto print_lan_done
:print_lan_on
echo   LAN     : on  (binds 0.0.0.0; no auth - trusted network only)
:print_lan_done
echo.

if not defined ENABLE_LAN goto start_server

rem --- read the WSL VM's IP (WSL2 NAT; changes per boot) ---
for /f "delims=" %%i in ('wsl -d %WSL_DISTRO% -- hostname -I') do set "WSL_IPS=%%i"
for /f "tokens=1" %%i in ("%WSL_IPS%") do set "WSL_IP=%%i"
if defined WSL_IP goto wsl_ip_ok
echo ERROR: could not read the WSL IP (is WSL2 / distro "%WSL_DISTRO%" available?).
pause
exit /b 1
:wsl_ip_ok
echo   WSL IP   : %WSL_IP%

rem --- read this PC's LAN IP (skip loopback, APIPA and the WSL vEthernet 172.16-31.x) ---
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$a = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^127\.' -and $_.IPAddress -notmatch '^169\.254\.' -and $_.IPAddress -notmatch '^172\.(1[6-9]|2[0-9]|3[01])\.' } | Select-Object -First 1; if ($a) { $a.IPAddress }"`) do set "LAN_IP=%%i"
if not defined LAN_IP set "LAN_IP=<this-PC-LAN-IP>"
echo   LAN IP   : %LAN_IP%  (devices use http://%LAN_IP%:%SERVE_PORT%/v1)

rem --- Windows-side forwarding: portproxy (WSL IP changes per reboot, so a
rem     stale forward from a previous run is deleted first) ---
netsh interface portproxy delete v4tov4 listenport=%SERVE_PORT% listenaddress=0.0.0.0 >nul 2>&1
netsh interface portproxy add v4tov4 listenport=%SERVE_PORT% listenaddress=0.0.0.0 connectport=%SERVE_PORT% connectaddress=%WSL_IP%
if errorlevel 1 goto portproxy_failed
echo   portproxy: 0.0.0.0:%SERVE_PORT% -^> %WSL_IP%:%SERVE_PORT%
goto portproxy_ok
:portproxy_failed
echo ERROR: could not create the portproxy rule. Check that the 'IP Helper'
echo (iphlpsvc) service is running, then retry.
pause
exit /b 1
:portproxy_ok

rem --- Windows Firewall: allow inbound TCP on the port ---
netsh advfirewall firewall delete rule name="Qwen3.8-27B API LAN %SERVE_PORT%" >nul 2>&1
netsh advfirewall firewall add rule name="Qwen3.8-27B API LAN %SERVE_PORT%" dir=in action=allow protocol=TCP localport=%SERVE_PORT% >nul 2>&1
if errorlevel 1 goto fw_failed
echo   firewall : inbound TCP %SERVE_PORT% allowed
goto fw_done
:fw_failed
echo Warning: could not add the firewall rule. LAN access may be blocked
echo if Windows Firewall is enabled.
:fw_done
echo.

:start_server
echo The server stays in this window. Press Ctrl-C to stop it; the
if defined ENABLE_LAN (
  echo portproxy and firewall rule are removed when it stops.
) else (
  echo window then stays open so you can read the result.
)
echo.

rem wsl runs as a child process here: the vLLM process stays inside it,
rem this window is the service console, and after wsl exits the batch
rem continues to the pause below (which keeps the window open).
wsl -d %WSL_DISTRO% -- bash %DIRECT_SH% start %LAN_FLAG% %ARGS%
set "RC=%ERRORLEVEL%"

if not defined ENABLE_LAN goto after_lan_cleanup
rem --- cleanup: remove the forward + firewall rule (the WSL IP may change on
rem     the next boot; a stale forward must not linger) ---
netsh interface portproxy delete v4tov4 listenport=%SERVE_PORT% listenaddress=0.0.0.0 >nul 2>&1
netsh advfirewall firewall delete rule name="Qwen3.8-27B API LAN %SERVE_PORT%" >nul 2>&1
:after_lan_cleanup

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
if defined ENABLE_LAN (
  echo LAN portproxy and firewall rule removed.
)
echo Press any key to close this window.
pause
