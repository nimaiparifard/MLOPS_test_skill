@echo off
REM Re-run the pipeline so MLflow gets the new runs and plots, then open the MLflow UI.
REM Usage: run_mlflow.bat          (full: split, tune, train, test, evaluate, register)
REM        run_mlflow.bat quick    (skip tuning, reuse best_params.json)
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if "%MLFLOW_TRACKING_URI%"=="" set MLFLOW_TRACKING_URI=http://localhost:5000

REM 1. make sure the MLflow server is up (docker compose service "mlflow")
curl -s -f %MLFLOW_TRACKING_URI%/health >nul 2>&1
if not errorlevel 1 goto :up
echo MLflow is not running at %MLFLOW_TRACKING_URI%, starting it with docker compose...
docker compose up -d mlflow
if errorlevel 1 goto :fail
set tries=0
:wait
timeout /t 3 /nobreak >nul
curl -s -f %MLFLOW_TRACKING_URI%/health >nul 2>&1
if not errorlevel 1 goto :up
set /a tries=tries+1
if %tries% LSS 20 goto :wait
echo MLflow did not start in time.
goto :fail

:up
echo MLflow is up at %MLFLOW_TRACKING_URI%

REM 2. pipeline (each step logs to MLflow)
python -m pipeline.create_dataset || goto :fail
if /i "%1"=="quick" goto :train
python -m pipeline.tuning_experiment || goto :fail
:train
python -m pipeline.train || goto :fail
python -m pipeline.test || goto :fail
python -m pipeline.evaluate || goto :fail
python -m pipeline.register_model || goto :fail

REM 3. open the UI
start "" %MLFLOW_TRACKING_URI%
echo Done. Plots are in the Artifacts tab of the runs in "tuning-experiment" and "classification-experiment".
exit /b 0

:fail
echo.
echo FAILED - see the error above.
exit /b 1
