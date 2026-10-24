@echo off
rem Walkthrough Sync launcher.
rem First run creates a local virtual environment and installs everything,
rem including PyTorch with CUDA (a few GB - it takes a while once).
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem A venv built on a Python that's too new for the AI libraries (3.15 has no
rem PyYAML wheels yet, for one) or left half-installed is rebuilt from scratch.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)" >nul 2>&1 || (
        echo Removing a virtual environment built on an unsupported Python version...
        rmdir /s /q .venv
    )
)

if exist ".venv\installed.ok" goto :launch

if not exist ".venv\Scripts\python.exe" (
    call :findpython || goto :nopython
    echo Creating virtual environment with !PYEXE!
    !PYEXE! -m venv .venv || goto :nopython
)

echo Installing dependencies - the first run downloads PyTorch, so this takes a few minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
echo.
echo Installing PyTorch with CUDA...
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 ^
  || ".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 ^
  || ".venv\Scripts\python.exe" -m pip install torch torchvision ^
  || goto :pipfail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :pipfail
".venv\Scripts\python.exe" -c "import torch; print('PyTorch', torch.__version__, '- GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU mode)')"
echo ok> ".venv\installed.ok"

:launch
start "" ".venv\Scripts\pythonw.exe" main.py
exit /b 0

rem ---------------------------------------------------------------------
rem Find a Python between 3.10 and 3.14 (PyTorch and friends lag behind the
rem newest release). Sets PYEXE.
:findpython
for %%P in (
    "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
    "%LOCALAPPDATA%\Python\pythoncore-3.13-64\python.exe"
    "%LOCALAPPDATA%\Python\pythoncore-3.12-64\python.exe"
    "C:\Python314\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
) do (
    if exist %%P (
        %%P -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)" >nul 2>&1 && (
            set PYEXE="%%~P"
            exit /b 0
        )
    )
)
for %%V in (3.14 3.13 3.12) do (
    py -%%V -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)" <nul >nul 2>&1 && (
        set PYEXE=py -%%V
        exit /b 0
    )
)
exit /b 1

:nopython
echo.
echo No Python between 3.10 and 3.14 was found. PyTorch doesn't support newer
echo versions yet - install Python 3.14 from python.org and run this again.
pause
exit /b 1

:pipfail
echo Dependency installation failed. See the messages above.
pause
exit /b 1
