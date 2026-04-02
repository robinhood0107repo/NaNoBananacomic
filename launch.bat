@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%src;%ROOT%.vendor;%ROOT%.bootstrap"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python is not available on PATH.
  echo Install Python 3.12 first, then run launch.bat again.
  exit /b 1
)

if "%~1"=="" (
  echo NaNoBananacomic launcher
  echo.
  echo This repository currently exposes the Step 1 CLI scaffold.
  echo.
  echo Example commands:
  echo   launch.bat init-project ^<PROJECT_FOLDER^>
  echo   launch.bat scan-pages ^<PROJECT_FOLDER^>
  echo   launch.bat detect ^<PROJECT_FOLDER^> --page-id 0001
  echo   launch.bat validate-step1 ^<PROJECT_FOLDER^> --page-id 0001
  echo.
  echo Note:
  echo   For actual detection runs, install the dependencies from pyproject.toml first.
  python -m comic_pipeline --help
  exit /b %errorlevel%
)

python -m comic_pipeline %*
exit /b %errorlevel%
