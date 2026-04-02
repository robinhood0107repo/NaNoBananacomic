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
  echo This repository currently exposes Step 1, Step 2, Step 3, and Step 4 CLI workflows.
  echo.
  echo Example commands:
  echo   launch.bat init-project ^<PROJECT_FOLDER^>
  echo   launch.bat scan-pages ^<PROJECT_FOLDER^>
  echo   launch.bat detect ^<PROJECT_FOLDER^> --page-id 0001
  echo   launch.bat validate-step1 ^<PROJECT_FOLDER^> --page-id 0001
  echo   launch.bat make-layer ^<PROJECT_FOLDER^> --page-id 0001
  echo   launch.bat make-handoff ^<PROJECT_FOLDER^> --page-id 0001
  echo   launch.bat import-external-result ^<PROJECT_FOLDER^> --page-id 0001 --input ^<RESULT_IMAGE^>
  echo   launch.bat validate-step3 ^<PROJECT_FOLDER^> --page-id 0001
  echo   launch.bat run-external-edit ^<PROJECT_FOLDER^> --page-id 0001
  echo   launch.bat restore-alpha ^<PROJECT_FOLDER^> --page-id 0001
  echo   launch.bat validate-step4 ^<PROJECT_FOLDER^> --page-id 0001
  echo.
  echo Note:
  echo   Step 3 API mode uses the GEMINI_API_KEY environment variable or a future GUI session key.
  python -m comic_pipeline --help
  exit /b %errorlevel%
)

python -m comic_pipeline %*
exit /b %errorlevel%
