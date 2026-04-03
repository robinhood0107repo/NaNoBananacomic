@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PYTHONPATH=%ROOT%src;%ROOT%.vendor;%ROOT%.bootstrap"
set "LOGDIR=%ROOT%logs\launcher"

if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>nul

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"
if not defined STAMP (
  set "STAMP=%DATE%_%TIME%"
  set "STAMP=%STAMP: =0%"
  set "STAMP=%STAMP:/=-%"
  set "STAMP=%STAMP::=-%"
  set "STAMP=%STAMP:.=-%"
  set "STAMP=%STAMP:,=-%"
)
set "LOGFILE=%LOGDIR%\launch_%STAMP%.log"

call :log INFO "NaNoBananacomic launcher started"
call :log INFO "Repository root: %ROOT%"

if not exist "%VENV%\Scripts\python.exe" (
  call :log INFO "Creating local virtual environment"
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.12 -m venv "%VENV%" >> "%LOGFILE%" 2>&1
    if errorlevel 1 call :fail "Failed to create .venv with py -3.12."
  ) else (
    where python >nul 2>nul
    if errorlevel 1 call :fail "Python 3.12 was not found. Install Python 3.12 and run launch.bat again."
    python -m venv "%VENV%" >> "%LOGFILE%" 2>&1
    if errorlevel 1 call :fail "Failed to create .venv with python -m venv."
  )
)

set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" call :fail "Local virtual environment is incomplete: %PY%"

call :log INFO "Checking runtime dependencies"
"%PY%" -c "import PySide6, cv2, numpy, PIL" >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  call :log INFO "Installing runtime dependencies from requirements.txt"
  "%PY%" -m pip install -r "%ROOT%requirements.txt" >> "%LOGFILE%" 2>&1
  if errorlevel 1 call :fail "Failed to install runtime dependencies."
)

if "%~1"=="" (
  call :log INFO "Launching NaNoBananacomic desktop app"
  set "COMIC_PIPELINE_GUI_LOGFILE=%LOGFILE%"
  "%PY%" -m comic_pipeline.gui.app >> "%LOGFILE%" 2>&1
  if errorlevel 1 call :fail "GUI exited with an error. See the launcher log below."
  exit /b 0
)

call :log INFO "Running CLI command: %*"
"%PY%" -m comic_pipeline %* >> "%LOGFILE%" 2>&1
if errorlevel 1 call :fail "CLI command failed. See the launcher log below."
exit /b 0

:log
echo [%~1] %~2
>> "%LOGFILE%" echo [%DATE% %TIME%] [%~1] %~2
exit /b 0

:show_log_tail
where powershell >nul 2>nul
if not errorlevel 1 (
  powershell -NoProfile -Command "Get-Content -Path '%LOGFILE%' -Tail 120"
  exit /b 0
)
type "%LOGFILE%"
exit /b 0

:fail
call :log ERROR "%~1"
echo.
echo [ERROR] %~1
echo [ERROR] Log file: %LOGFILE%
echo.
call :show_log_tail
echo.
pause
exit /b 1
