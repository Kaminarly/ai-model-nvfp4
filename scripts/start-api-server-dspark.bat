@echo off
chcp 65001 >nul 2>&1
setlocal EnableExtensions
title Qwen3.8-27B API Server - DSpark (SGLang + Docker)

rem =====================================================================
rem  start-api-server-dspark.bat - double-click Windows launcher for
rem  scripts/sglang-dspark.sh: runs the certified SGLang image in Docker
rem  inside WSL2 Ubuntu and keeps the service in THIS console window, so
rem  SGLang's load progress and request log are visible here and Ctrl-C
rem  stops the container. After the server stops it runs "wsl --shutdown"
rem  to fully shut down the WSL VM and release its GPU VRAM.
rem
rem  !! THE IMAGE IS NO LONGER ON THIS MACHINE (removed 2026-09-19) !!
rem  lmsysorg/sglang:qwen38-27b was deleted to free 41.9 GB of docker disk.
rem  The preflight below fails closed with
rem    image 'lmsysorg/sglang:qwen38-27b' is not present locally.
rem    fix: pull it (about 18 GB): docker pull lmsysorg/sglang:qwen38-27b
rem  Nothing is downloaded automatically - re-pull explicitly to use this
rem  route again. The vLLM and SparkInfer launchers are unaffected (their
rem  images/venvs are separate). Record and measurements:
rem  result/sglang-image-removal-result.md, README 4.8 / Q17.
rem
rem  This is the speculative-decoding path: SGLang + the DSpark draft
rem  (1.8x decode throughput at the same engine). It is NOT a vLLM
rem  launcher, but it serves on the same fixed port 8192 as
rem  start-api-server-vllm.bat / -mtp / -gguf, so clients keep one endpoint;
rem  only one engine runs at a time (the 27B model holds the VRAM).
rem
rem  The container is deliberately run in the FOREGROUND with no restart
rem  policy: WSL reclaims the distro as soon as its last session ends
rem  (InitTerminateInstanceInternal -> reboot(RB_POWER_OFF)), so a
rem  detached container is killed anyway and a --restart policy only turns
rem  that teardown into a reload/kill loop. Keeping the window open IS the
rem  keep-alive; Ctrl-C is the stop button.
rem
rem  Before the service starts, this launcher asks whether to enable LAN
rem  access: 1 enable / 2 disable (default) / 0 quit. Enable needs no
rem  change on the SGLang side (the container already binds 0.0.0.0 inside
rem  WSL) - it adds the Windows portproxy + firewall inbound rule, which
rem  are what actually expose the port beyond this PC, and re-launches
rem  elevated via UAC when needed. An internal --lan-enabled argument
rem  skips the menu after elevation.
rem
rem  Overridable environment variables:
rem    MODEL_DIR   WSL path of the target model folder (default below)
rem    DRAFT_DIR   WSL path of the DSpark draft folder (default below)
rem    WSL_DISTRO  WSL distribution name (default Ubuntu)
rem    SERVE_PORT  port (default 8192, fixed by convention across launchers)
rem    SGLANG_IMAGE / SGLANG_NAME   image tag / container name
rem  Command-line options are forwarded to sglang-dspark.sh, e.g.
rem  --context-length 65536, --no-spec, --dry-run.
rem
rem  The WSL side runs as root: docker needs it on this machine because user
rem  kami is not in the docker group. That only affects who runs docker -
rem  both model folders are mounted read-only.
rem =====================================================================

rem --- defaults (edit here or set the env vars above) ---
if not defined MODEL_DIR set "MODEL_DIR=/home/kami/models/Qwen3.8-27B-NVFP4-DSpark"
if not defined DRAFT_DIR set "DRAFT_DIR=/home/kami/models/Qwen3.8-27B-DSpark-Acc"
if not defined WSL_DISTRO set "WSL_DISTRO=Ubuntu"
if not defined SERVE_PORT set "SERVE_PORT=8192"
if not defined SGLANG_IMAGE set "SGLANG_IMAGE=lmsysorg/sglang:qwen38-27b"
if not defined SGLANG_NAME set "SGLANG_NAME=qwen38-sglang"

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

rem --- locate the project's scripts/sglang-dspark.sh ---
rem The launcher may sit anywhere (e.g. a copy on the Desktop). Try, in
rem order: an explicit PROJECT_DIR, this script's own folder upwards, then
rem the default project location on this machine.
if not defined PROJECT_DIR set "PROJECT_DIR=D:\Code\MJ-Project\ai-model-nvfp4"
set "PROJ=%PROJECT_DIR%"
if exist "%PROJ%\scripts\sglang-dspark.sh" goto found_launcher
set "PROJ=%SCRIPT_WIN%"
:find_launcher
if exist "%PROJ%\scripts\sglang-dspark.sh" goto found_launcher
set "PARENT=%PROJ%"
for %%I in ("%PROJ%\.") do set "PROJ=%%~dpI"
set "PROJ=%PROJ:~0,-1%"
if "%PROJ%"=="%PARENT%" goto no_launcher
goto find_launcher
:no_launcher
echo ERROR: scripts\sglang-dspark.sh not found.
echo Looked at: %PROJECT_DIR% and every folder above %SCRIPT_WIN%.
echo Set PROJECT_DIR to the project folder if it lives elsewhere, e.g.:
echo   set PROJECT_DIR=D:\Code\MJ-Project\ai-model-nvfp4
echo.
echo Press any key to close this window.
pause
exit /b 1
:found_launcher

rem Drive letter comes from the PROJECT path (the launcher may sit on a
rem different drive than the project, e.g. a Desktop copy on C:). WSL
rem mounts as lowercase (/mnt/d), so the letter is lowercased.
set "DRIVE=%PROJ:~0,1%"
for %%D in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
  if /I "%DRIVE%"=="%%D" set "DRIVE=%%D"
)
set "SGLANG_WIN=%PROJ%\scripts\sglang-dspark.sh"
set "SGLANG_SH=%SGLANG_WIN:\=/%"
set "SGLANG_SH=/mnt/%DRIVE%/%SGLANG_SH:~3%"
rem wsl.exe does not strip quotes from -d, and joins everything after --
rem into one command line parsed by bash, so the path must not be quoted;
rem spaces in it are escaped for bash instead.
set "SGLANG_SH=%SGLANG_SH: =\ %"

rem --- forward command-line options; default the two model dirs when absent ---
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
echo %ARGS% | findstr /C:"--draft-dir" >nul || set "ARGS=%ARGS% --draft-dir %DRAFT_DIR%"

set "LAN_FLAG="
if defined ENABLE_LAN set "LAN_FLAG=--lan"

rem --- share the optional env vars with WSL (WSLENV) ---
set "WSLENV=SERVE_PORT/u:SGLANG_IMAGE/u:SGLANG_NAME/u"

echo Starting the Qwen3.8-27B DSpark API server via WSL (%WSL_DISTRO%)...
echo   launcher: %SGLANG_SH%
echo   image   : %SGLANG_IMAGE%  (container %SGLANG_NAME%)
echo   model   : %MODEL_DIR%
echo   draft   : %DRAFT_DIR%
echo   options : %LAN_FLAG% %ARGS%
echo   port    : %SERVE_PORT%
if defined ENABLE_LAN goto print_lan_on
echo   LAN     : off (localhost only)
goto print_lan_done
:print_lan_on
echo   LAN     : on  (no auth - trusted network only)
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
netsh advfirewall firewall delete rule name="Qwen3.8-27B DSpark API LAN %SERVE_PORT%" >nul 2>&1
netsh advfirewall firewall add rule name="Qwen3.8-27B DSpark API LAN %SERVE_PORT%" dir=in action=allow protocol=TCP localport=%SERVE_PORT% >nul 2>&1
if errorlevel 1 goto fw_failed
echo   firewall : inbound TCP %SERVE_PORT% allowed
goto fw_done
:fw_failed
echo Warning: could not add the firewall rule. LAN access may be blocked
echo if Windows Firewall is enabled.
:fw_done
echo.

:start_server
echo The server stays in this window. Press Ctrl-C to stop it (SGLang drains
echo the in-flight request first); loading takes about a minute and the ready
echo line is "The server is fired up and ready to roll!".
if defined ENABLE_LAN (
  echo portproxy and firewall rule are removed when it stops.
) else (
  echo window then stays open so you can read the result.
)
echo.

rem wsl runs as a child process here: the container stays inside it, this
rem window is the service console, and after wsl exits the batch continues
rem to the cleanup below. -u root is required for docker on this machine.
wsl -d %WSL_DISTRO% -u root -- bash %SGLANG_SH% start %LAN_FLAG% %ARGS%
set "RC=%ERRORLEVEL%"

if not defined ENABLE_LAN goto after_lan_cleanup
rem --- cleanup: remove the forward + firewall rule (the WSL IP may change on
rem     the next boot; a stale forward must not linger) ---
netsh interface portproxy delete v4tov4 listenport=%SERVE_PORT% listenaddress=0.0.0.0 >nul 2>&1
netsh advfirewall firewall delete rule name="Qwen3.8-27B DSpark API LAN %SERVE_PORT%" >nul 2>&1
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
echo Note: a stopped container is kept for post-mortem; inspect it later with
echo   wsl -d %WSL_DISTRO% -u root -- docker logs %SGLANG_NAME%
echo The 2026-09-19 qwen38-sglang container was removed together with the
echo image; its last log is archived inside WSL at
echo   ~/logs/qwen38-sglang-2026-09-19.log
echo Press any key to close this window.
pause
