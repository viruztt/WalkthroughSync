@echo off
rem Adds a Startup shortcut so Walkthrough Sync launches when you sign in to Windows.
rem Run it again after moving the folder; run remove_startup.bat to undo.
setlocal
cd /d "%~dp0"

set "LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Walkthrough Sync.lnk"
set "TARGET=%CD%\.venv\Scripts\pythonw.exe"
set "SCRIPT=%CD%\main.py"
set "WORKDIR=%CD%"

if not exist "%TARGET%" (
    echo The virtual environment is missing.
    echo Run run.bat once first - it creates .venv and installs dependencies.
    pause
    exit /b 1
)

powershell -NoProfile -NonInteractive -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK);" ^
  "$s.TargetPath = $env:TARGET;" ^
  "$s.Arguments = '\"' + $env:SCRIPT + '\"';" ^
  "$s.WorkingDirectory = $env:WORKDIR;" ^
  "$s.IconLocation = $env:TARGET;" ^
  "$s.Description = 'Walkthrough Sync';" ^
  "$s.Save()"

if exist "%LNK%" (
    echo.
    echo Installed - Walkthrough Sync will start with Windows.
) else (
    echo.
    echo Could not create the shortcut. See any message above.
)
pause
