@echo off
setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PYTHONPATH=%ROOT%src;%ROOT%.vendor;%ROOT%.bootstrap"

if not exist "%VENV%\Scripts\python.exe" (
  echo [INFO] Creating local virtual environment...
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.12 -m venv "%VENV%"
  ) else (
    where python >nul 2>nul
    if errorlevel 1 (
      echo [ERROR] Python 3.12 was not found.
      echo Install Python 3.12 for Windows, then run launch.bat again.
      exit /b 1
    )
    python -m venv "%VENV%"
  )
)

set "PY=%VENV%\Scripts\python.exe"
set "PYW=%VENV%\Scripts\pythonw.exe"

if not exist "%PY%" (
  echo [ERROR] Local virtual environment is incomplete: "%PY%"
  exit /b 1
)

"%PY%" -c "import PySide6, cv2, numpy, PIL" >nul 2>nul
if errorlevel 1 (
  echo [INFO] Installing runtime dependencies...
  "%PY%" -m pip install -r "%ROOT%requirements.txt"
  if errorlevel 1 (
    echo [ERROR] Failed to install runtime dependencies.
    exit /b 1
  )
)

if "%~1"=="" (
  echo [INFO] Launching NaNoBananacomic desktop app...
  if exist "%PYW%" (
    start "" "%PYW%" -m comic_pipeline.gui.app
    exit /b 0
  )
  "%PY%" -m comic_pipeline.gui.app
  exit /b %errorlevel%
)

"%PY%" -m comic_pipeline %*
exit /b %errorlevel%
