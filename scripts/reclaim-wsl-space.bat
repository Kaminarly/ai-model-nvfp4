@echo off
setlocal EnableExtensions
title Reclaim D: space from the WSL ext4.vhdx (diskpart compact)

rem =====================================================================
rem  reclaim-wsl-space.bat - shrink D:\WSL\ext4.vhdx back to its real size
rem  after large deletions inside the WSL filesystem (e.g. removing a big
rem  docker image such as lmsysorg/sglang:qwen38-27b, 41.9 GB).
rem
rem  Why a script is needed at all: a NON-SPARSE WSL vhdx never returns
rem  freed blocks to Windows on its own. Deleting files inside the distro
rem  (or `fstrim`) only frees them inside ext4. The no-admin alternative,
rem  `wsl --manage <distro> --set-sparse true`, is REFUSED by Microsoft for
rem  an existing distro ("sparse VHD support is currently disabled due to
rem  potential data corruption"); forcing it requires --allow-unsafe, which
rem  this script deliberately does not use. Offline compaction of the
rem  STOPPED vhdx is the documented, non-destructive path: the volume is
rem  attached READ-ONLY, compacted, then detached.
rem
rem  MUST run as administrator (diskpart). Double-click it and accept the
rem  UAC prompt, or run it from an already-elevated prompt.
rem
rem  WARNING: `wsl --shutdown` runs first, so every WSL session (and any
rem  vLLM / SparkInfer service running inside WSL) is stopped. Nothing
rem  inside the distro is modified by the compaction itself.
rem
rem  Overridable environment variables:
rem    VHDX    path of the WSL virtual disk (default below)
rem    DISTRO  WSL distribution to restart afterwards (default Ubuntu)
rem =====================================================================

rem --- defaults ---
if not defined VHDX set "VHDX=D:\WSL\ext4.vhdx"
if not defined DISTRO set "DISTRO=Ubuntu"

rem --- administrator check (diskpart needs it); same probe as the launchers ---
fsutil dirty query %SystemDrive% >nul 2>&1
if errorlevel 1 goto need_admin
goto admin_ok

:need_admin
echo Requesting administrator rights via UAC...
powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
echo.
echo The elevated copy continues in a new window. If no UAC prompt appears
echo or you denied it, right-click this file and pick "Run as administrator".
echo Press any key to close this window.
pause
exit /b 1

:admin_ok
if not exist "%VHDX%" goto no_vhdx

set "FREE_BEFORE="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[math]::Round((Get-PSDrive D).Free/1GB,2)"`) do set "FREE_BEFORE=%%i"
set "SIZE_BEFORE="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[math]::Round((Get-Item '%VHDX%').Length/1GB,2)"`) do set "SIZE_BEFORE=%%i"
echo.
echo   vhdx          : %VHDX%
echo   logical size  : %SIZE_BEFORE% GiB
echo   D: free before: %FREE_BEFORE% GiB
echo.

echo Stopping WSL so the virtual disk is not in use...
wsl --shutdown
powershell -NoProfile -Command "Start-Sleep -Seconds 5"

rem --- compact with diskpart (read-only attach: the distro is untouched) ---
set "DPSCRIPT=%TEMP%\wsl-vhdx-compact.txt"
>  "%DPSCRIPT%" echo select vdisk file="%VHDX%"
>> "%DPSCRIPT%" echo attach vdisk readonly
>> "%DPSCRIPT%" echo compact vdisk
>> "%DPSCRIPT%" echo detach vdisk
>> "%DPSCRIPT%" echo exit
echo Compacting - this can take several minutes and shows diskpart output below.
echo.
diskpart /s "%DPSCRIPT%"
set "DPRC=%ERRORLEVEL%"
del "%DPSCRIPT%" >nul 2>&1
echo.
if not "%DPRC%"=="0" goto compact_failed

set "FREE_AFTER="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[math]::Round((Get-PSDrive D).Free/1GB,2)"`) do set "FREE_AFTER=%%i"
set "SIZE_AFTER="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[math]::Round((Get-Item '%VHDX%').Length/1GB,2)"`) do set "SIZE_AFTER=%%i"
echo   logical size : %SIZE_BEFORE% GiB  -^>  %SIZE_AFTER% GiB
echo   D: free      : %FREE_BEFORE% GiB  -^>  %FREE_AFTER% GiB
echo.

echo Starting WSL (%DISTRO%) again and checking it boots...
wsl -d %DISTRO% -- echo WSL is back
if errorlevel 1 goto boot_failed
echo Done. VRAM held by WSL was released by the shutdown as well.
echo Press any key to close this window.
pause
exit /b 0

:no_vhdx
echo ERROR: virtual disk not found: %VHDX%
echo Set VHDX to the right path, e.g.:
echo   set VHDX=D:\WSL\ext4.vhdx
echo Press any key to close this window.
pause
exit /b 1

:compact_failed
echo ERROR: diskpart returned %DPRC%. The virtual disk was detached; nothing
echo was modified. Start WSL again with: wsl -d %DISTRO% -- echo ok
echo Press any key to close this window.
pause
exit /b 1

:boot_failed
echo WARNING: WSL did not report a successful boot. The disk was compacted
echo but check the distribution manually: wsl -l -v
echo Press any key to close this window.
pause
exit /b 1