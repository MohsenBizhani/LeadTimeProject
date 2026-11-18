@echo off
setlocal EnableDelayedExpansion
title Lead Time Prediction Platform Runner
echo ========================================================
echo Lead Time Prediction Platform Runner
echo ========================================================
echo.

set "ROOT=%~dp0"
set "LOG_DIR=%ROOT%logs"
set "LOG_FILE=%LOG_DIR%\backend.log"
set "PID_FILE=%LOG_DIR%\backend.pid"
set "MAX_RETRY=3"
set "RETRY_COUNT=0"
set "MAX_WAIT=15"
set "REQD=fastapi uvicorn catboost pydantic pandas numpy scikit-learn joblib python-multipart"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" 2>NUL

:: ---------- Pre-flight: required project files exist ----------
if not exist "%ROOT%app\backend\main.py" goto err_backend
if not exist "%ROOT%app\backend\serve.py" goto err_backend
if not exist "%ROOT%app\frontend\index.html" goto err_frontend

:: ---------- Locate a Python interpreter ----------
echo Locating Python...
set "PYTHON_EXE="
set "PYTHON_EXE_FALLBACK="
call :consider python
call :consider py -3
call :consider "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :consider "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
call :consider "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
call :consider "%ProgramFiles%\Python312\python.exe"
call :consider "%ProgramFiles%\Python311\python.exe"
call :consider "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"

if defined PYTHON_EXE goto python_ready
if defined PYTHON_EXE_FALLBACK (
    set "PYTHON_EXE=%PYTHON_EXE_FALLBACK%"
    goto python_ready
)

:: ---------- Auto-Install Python 3.10 if missing ----------
echo.
echo Python 3.10 was not found on this system.
echo Downloading and installing Python 3.10 automatically (requires internet)...
echo.

set "PY_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
set "PY_INSTALLER=%TEMP%\python_310_installer.exe"

powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER%'"
if not exist "%PY_INSTALLER%" goto err_python_install

echo Running silent installation of Python 3.10 (Adding to PATH)...
"%PY_INSTALLER%" /quiet InstallAllUsers=0 TargetDir="%LOCALAPPDATA%\Programs\Python\Python310" PrependPath=1 Include_test=0

:: Clean up installer
del "%PY_INSTALLER%" >NUL 2>&1

:: Re-verify Python location after auto-install
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto python_ready
)

:: Try default python command post-install
call :consider python
if defined PYTHON_EXE goto python_ready

goto err_python

:python_ready
echo Using Python: "%PYTHON_EXE%"
for %%D in ("%PYTHON_EXE%") do set "PYDIR=%%~dpD"
set "PYTHONW_EXE=%PYTHON_EXE%"
if exist "%PYDIR%pythonw.exe" set "PYTHONW_EXE=%PYDIR%pythonw.exe"
echo.

:: ---------- Ensure dependencies ----------
:check_dependencies
"%PYTHON_EXE%" -c "import fastapi, uvicorn, catboost, pydantic, pandas, numpy, sklearn, joblib, multipart" 2>NUL
if not errorlevel 1 goto deps_ok
set /a RETRY_COUNT+=1
if !RETRY_COUNT! GTR %MAX_RETRY% goto err_deps
echo Installing required packages (attempt !RETRY_COUNT! of %MAX_RETRY%)...
"%PYTHON_EXE%" -m pip install --upgrade pip >NUL 2>&1
"%PYTHON_EXE%" -m pip install %REQD%
goto check_dependencies

:deps_ok
echo.
echo All required packages are available.
echo.

:: ---------- Port probe: is our backend (or something else) on 8001? ----------
echo Checking port 8001...
set "PROBE=NOT_READY"
curl.exe -s -m 3 http://127.0.0.1:8001/health | findstr /C:"models_loaded" >NUL 2>&1
if not errorlevel 1 set "PROBE=READY_OURS"
echo (probe says: %PROBE%)
if "%PROBE%"=="READY_OURS" goto port_ours
curl.exe -s -m 3 -o NUL http://127.0.0.1:8001/ 2>NUL
if not errorlevel 1 set "PROBE=READY_OTHER"
if "%PROBE%"=="READY_OTHER" goto port_other
goto start_server

:port_ours
echo Our backend is already running on port 8001 - reusing it.
echo Running this script again is safe and will not start a second server.
echo.
goto resolve_pid

:port_other
echo.
echo ERROR: Port 8001 is occupied by a different application, not our backend.
echo The process currently holding the port:
netstat -ano | findstr "LISTENING" | findstr ":8001 "
echo.
echo Fix: stop that program, or free port 8001, then run this script again.
echo.
pause
exit /b 1

:: ---------- Fresh start: launch detached, windowless, with logging ----------
:start_server
echo Starting the backend on port 8001 (detached background process)...
echo Log file: %LOG_FILE%
echo [%date% %time%] Launching backend on port 8001 >> "%LOG_FILE%"
start "LeadTimeBackend" /D "%ROOT%app\backend" "%PYTHONW_EXE%" serve.py

:: ---------- Wait for healthy startup (models must load) ----------
echo Waiting for the backend to start and load the models...
set /a STARTUP_WAIT=0

:check_server
curl.exe -s -m 3 http://127.0.0.1:8001/health | findstr /C:"models_loaded" | findstr /C:":true" >NUL 2>&1
if not errorlevel 1 goto server_ok
set /a STARTUP_WAIT+=1
if !STARTUP_WAIT! GEQ %MAX_WAIT% goto server_fail
echo Waiting for server... ^(!STARTUP_WAIT!/%MAX_WAIT%^)
ping -n 3 127.0.0.1 >NUL 2>&1
goto check_server

:server_fail
echo.
echo ERROR: The backend did not become ready in time.
echo.
echo ----- last 40 lines of the backend log -----
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath '%LOG_FILE%' -Tail 40"
echo ----- end of log -----
echo.
echo Common causes:
echo   - Model files missing under lead_time_platform_final\platform\artifacts_fixed
echo   - Port 8001 taken by another program, check with:  netstat -ano ^| findstr 8001
echo   - A Python package failed to install, see the messages above
echo.
echo You can also start the backend manually to see all errors:
echo   cd /d "%ROOT%app\backend"
echo   "%PYTHON_EXE%" -m uvicorn main:app --host 127.0.0.1 --port 8001
echo.
pause
exit /b 1

:server_ok
echo.
echo The backend is ready and the models are loaded.
echo.

:: ---------- Resolve backend PID so the user can stop it later ----------
:resolve_pid
set "BACKEND_PID="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":8001 "') do set "BACKEND_PID=%%a"
if defined BACKEND_PID goto pid_found
echo WARNING: The backend is running but its PID could not be detected via netstat.
echo To find and stop it later, run:  netstat -ano ^| findstr 8001
goto open_ui

:pid_found
echo !BACKEND_PID!> "%PID_FILE%"
echo Backend PID: !BACKEND_PID!  (saved to %PID_FILE%)
echo.

:open_ui
echo Opening the dashboard in your default browser...
start "" "%ROOT%app\frontend\index.html"

echo.
echo Platform is running.
echo Dashboard file: app\frontend\index.html
echo Backend API Docs: http://127.0.0.1:8001/docs
echo Backend Health:   http://127.0.0.1:8001/health
echo Backend log:      %LOG_FILE%
echo.
echo The server keeps running in the background after you close this window.
echo To stop it later:
if defined BACKEND_PID echo   taskkill /PID !BACKEND_PID! /F
if not defined BACKEND_PID echo   netstat -ano ^| findstr 8001
echo.
echo Keep this window open while you use the platform, or press any key to close it.
pause
goto :eof

:: ---------- Subroutine: test a candidate interpreter ----------
:consider
if defined PYTHON_EXE goto :eof
set "CAND="
%* -c "import sys,os;open(os.path.join(os.environ['TEMP'],'lt_py.txt'),'w').write(sys.executable)" >NUL 2>&1
if exist "%TEMP%\lt_py.txt" set /p CAND=<"%TEMP%\lt_py.txt"
del "%TEMP%\lt_py.txt" >NUL 2>&1
if not defined CAND goto :eof
if not defined PYTHON_EXE_FALLBACK set "PYTHON_EXE_FALLBACK=%CAND%"
%* -c "import fastapi, uvicorn, catboost, pydantic, pandas, numpy, sklearn, joblib, multipart" >NUL 2>&1
if not errorlevel 1 set "PYTHON_EXE=%CAND%"
goto :eof

:: ---------- Error messages ----------
:err_python
echo.
echo ERROR: Python 3.10 could not be found or automatically installed.
echo Please install Python 3.10 manually from https://www.python.org/downloads/
echo and ensure "Add python.exe to PATH" is ticked during installation.
echo.
pause
exit /b 1

:err_python_install
echo.
echo ERROR: Failed to download and install Python 3.10 automatically.
echo Please check your internet connection or download it manually from https://www.python.org/downloads/
echo.
pause
exit /b 1

:err_backend
echo.
echo ERROR: Required program files are missing.
echo   Expected: %ROOT%app\backend\main.py  and  %ROOT%app\backend\serve.py
echo Fix: extract the whole ZIP again, keeping the folder structure, and run
echo      run_platform.bat from inside the extracted folder.
echo.
pause
exit /b 1

:err_frontend
echo.
echo ERROR: The dashboard file is missing.
echo   Expected: %ROOT%app\frontend\index.html
echo Fix: extract the whole ZIP again, keeping the folder structure.
echo.
pause
exit /b 1

:err_deps
echo.
echo ERROR: Could not install the required packages after %MAX_RETRY% attempts.
echo This step needs an internet connection on the first run.
echo Try manually and read the error message:
echo   "%PYTHON_EXE%" -m pip install %REQD%
echo.
pause
exit /b 1