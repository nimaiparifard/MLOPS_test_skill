# Airflow DAG for the classification (credit risk / fraud) pipeline.
#
# Every task runs one stage of dvc.yaml with `dvc repro -s <stage>`, so Airflow does the
# scheduling / retries / UI and DVC keeps doing the data versioning (stages whose inputs did
# not change are skipped by DVC).
#
#   check_mlflow -> generate_data -> create_dataset -> tune -> train -> test -> evaluate
#                                                                      -> register_model -> dvc_push
#
# Env vars (all optional): PIPELINE_DIR, MLFLOW_TRACKING_URI, PIPELINE_PYTHON, DVC_REMOTE.
# Needs: the MLflow server (docker compose up -d mlflow) and, for dvc_push, the S3 container.
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2
    from airflow.decorators import dag, task

# the folder with dvc.yaml (this file may be copied to the Airflow dags/ folder, so allow an override)
PROJECT_DIR = Path(os.environ.get("PIPELINE_DIR", Path(__file__).resolve().parent.parent))
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
# python that has dvc/mlflow/optuna installed (the Docker image uses a separate venv for it)
PIPELINE_PYTHON = os.environ.get("PIPELINE_PYTHON", sys.executable)
# optional: name of the DVC remote to push to (the container needs a different S3 address)
DVC_REMOTE = os.environ.get("DVC_REMOTE")


def run(*cmd):
    """Run a command inside the project folder and fail the task if it fails."""
    env = {**os.environ, "PYTHONUTF8": "1"}  # MLflow prints emoji; avoid Windows encoding errors
    # dvc.yaml calls plain `python`: make that the pipeline's python, not Airflow's own
    env["PATH"] = str(Path(PIPELINE_PYTHON).parent) + os.pathsep + env.get("PATH", "")
    print(f"$ {' '.join(cmd)}  (in {PROJECT_DIR})")
    subprocess.run(cmd, cwd=PROJECT_DIR, env=env, check=True)


def run_stage(stage):
    run(PIPELINE_PYTHON, "-m", "dvc", "repro", "--single-item", stage)


@dag(
    dag_id="classification_pipeline",
    description="generate data -> split -> tune -> train -> test -> evaluate -> register model",
    start_date=datetime(2026, 1, 1),
    schedule=None,  # run by hand; e.g. use "@daily" or "0 3 * * 1" to schedule it
    catchup=False,
    max_active_runs=1,  # the stages share files, so never run two pipelines at once
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["ml", "dvc", "mlflow"],
)
def pipeline():
    @task
    def check_mlflow():
        # fail early (before any work) if the tracking server is down
        with urllib.request.urlopen(f"{MLFLOW_URI}/health", timeout=10) as resp:
            print(f"MLflow at {MLFLOW_URI}: {resp.read().decode()}")

    @task
    def generate_data():
        run_stage("generate_data")  # synthetic credit / fraud tables

    @task
    def create_dataset():
        run_stage("create_dataset")  # train / val / test split

    @task
    def tune():
        run_stage("tune")  # Optuna hyperparameter search, tracked in MLflow

    @task
    def train():
        run_stage("train")

    @task
    def test():
        run_stage("test")

    @task
    def evaluate():
        run_stage("evaluate")

    @task
    def register_model():
        run_stage("register_model")

    @task
    def dvc_push():
        run(PIPELINE_PYTHON, "-m", "dvc", "push", *(["-r", DVC_REMOTE] if DVC_REMOTE else []))

    # --- order of the tasks ---
    (
        check_mlflow()
        >> generate_data()
        >> create_dataset()
        >> tune()
        >> train()
        >> test()
        >> evaluate()
        >> register_model()
        >> dvc_push()
    )


pipeline()
