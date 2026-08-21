@echo off
chcp 65001 >nul 2>&1
setlocal EnableExtensions
title Qwen3.6-27B Fable-Fusion GGUF API Server (llama.cpp, 128k)

rem =====================================================================
rem  start-api-server-gguf.bat - double-click Windows launcher for the
rem  llama.cpp llama-server running the GGUF model:
rem    Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q5_K_M.gguf
rem  with the settings verified on this machine (RTX 5090):
rem    - context 128000, KV cache quantized to q8_0
rem    - concurrent slots auto, 8 CPU threads, full GPU offload
rem      + Flash Attention (from ~/llama.cpp/llama-server.sh)
rem    - sampling defaults: temperature 0.7, top-k 20, top-p 0.8, min-p 0,
rem      repeat-penalty 1.0, presence-penalty 1.5
rem    - reasoning OFF (answers directly, no thinking)
rem    - port 8192, loopback only (127.0.0.1) unless LAN is enabled
rem  Before the service starts, this launcher asks whether to enable LAN
rem  access: 1 enable / 2 disable (default) / 0 quit. Enable binds
rem  0.0.0.0, then does Windows portproxy + firewall (UAC if needed).
rem  An internal --lan-enabled argument skips the menu after elevation.
rem  The service runs in THIS console window (Ctrl-C stops it); after it
rem  stops, "wsl --shutdown" fully shuts down the WSL VM to release VRAM.
rem  This is the llama.cpp stack, NOT the vLLM stack - do not run it at
rem  the same time as start-api-server.bat / start-api-server-mtp.bat
rem  (all three default to port 8192).
rem
rem  Overridable environment variables:
rem    MODEL_GGUF   WSL path of the .gguf file (default below)
rem    LLAMA_PORT   port (default 8192)
rem    LLAMA_CTX    context length (default 128000)
rem    WSL_DISTRO   WSL distribution name (default Ubuntu)
rem =====================================================================

rem --- defaults (edit here or set the env vars above) ---
if not defined MODEL_GGUF set "MODEL_GGUF=/home/kami/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q5_K_M.gguf"
if not defined LLAMA_PORT set "LLAMA_PORT=8192"
if not defined LLAMA_CTX set "LLAMA_CTX=128000"
if not defined WSL_DISTRO set "WSL_DISTRO=Ubuntu"

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

rem --- llama-server launcher inside WSL (fixed location) ---
set "LLAMA_SH=/home/kami/llama.cpp/llama-server.sh"

rem --- the tuned llama-server arguments (see header) ---
set "LLAMA_ARGS=-m %MODEL_GGUF% -c %LLAMA_CTX% -ctk q8_0 -ctv q8_0 -t 8 --temp 0.7 --top-k 20 --top-p 0.8 --min-p 0 --repeat-penalty 1.0 --presence-penalty 1.5 --reasoning off --alias Qwen3.6-27B-Fable-Fusion --port %LLAMA_PORT%"
rem llama-server.sh hard-codes --host 127.0.0.1 then appends "$@"; a later
rem --host 0.0.0.0 overrides it (llama.cpp last-write-wins).
if defined ENABLE_LAN set "LLAMA_ARGS=%LLAMA_ARGS% --host 0.0.0.0"

echo Starting the Qwen3.6-27B GGUF API server via WSL (%WSL_DISTRO%)...
echo   launcher: %LLAMA_SH%
echo   model   : %MODEL_GGUF%
echo   args    : %LLAMA_ARGS%
echo   port    : %LLAMA_PORT%
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
echo   LAN IP   : %LAN_IP%  (devices use http://%LAN_IP%:%LLAMA_PORT%/v1)

rem --- Windows-side forwarding: portproxy (WSL IP changes per reboot, so a
rem     stale forward from a previous run is deleted first) ---
netsh interface portproxy delete v4tov4 listenport=%LLAMA_PORT% listenaddress=0.0.0.0 >nul 2>&1
netsh interface portproxy add v4tov4 listenport=%LLAMA_PORT% listenaddress=0.0.0.0 connectport=%LLAMA_PORT% connectaddress=%WSL_IP%
if errorlevel 1 goto portproxy_failed
echo   portproxy: 0.0.0.0:%LLAMA_PORT% -^> %WSL_IP%:%LLAMA_PORT%
goto portproxy_ok
:portproxy_failed
echo ERROR: could not create the portproxy rule. Check that the 'IP Helper'
echo (iphlpsvc) service is running, then retry.
pause
exit /b 1
:portproxy_ok

rem --- Windows Firewall: allow inbound TCP on the port ---
netsh advfirewall firewall delete rule name="Qwen3.6-27B GGUF API LAN %LLAMA_PORT%" >nul 2>&1
netsh advfirewall firewall add rule name="Qwen3.6-27B GGUF API LAN %LLAMA_PORT%" dir=in action=allow protocol=TCP localport=%LLAMA_PORT% >nul 2>&1
if errorlevel 1 goto fw_failed
echo   firewall : inbound TCP %LLAMA_PORT% allowed
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

rem wsl runs as a child process here: llama-server stays inside it, this
rem window is the service console, and after wsl exits the batch continues
rem to the pause below (which keeps the window open).
wsl -d %WSL_DISTRO% -- bash %LLAMA_SH% %LLAMA_ARGS%
set "RC=%ERRORLEVEL%"

if not defined ENABLE_LAN goto after_lan_cleanup
rem --- cleanup: remove the forward + firewall rule (the WSL IP may change on
rem     the next boot; a stale forward must not linger) ---
netsh interface portproxy delete v4tov4 listenport=%LLAMA_PORT% listenaddress=0.0.0.0 >nul 2>&1
netsh advfirewall firewall delete rule name="Qwen3.6-27B GGUF API LAN %LLAMA_PORT%" >nul 2>&1
:after_lan_cleanup

rem The API server has stopped (Ctrl-C or otherwise). Fully shut down the
rem WSL VM so the GPU VRAM it held is released back to Windows.
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
