# RiskLens ML Pipeline

[![CI](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/ci.yml)

A reproducible machine-learning pipeline for **credit-default risk** and **card-fraud** classification.
It generates synthetic data, tunes five model families with Optuna, trains the best one, evaluates it on a
time-based test set, and registers it in the MLflow Model Registry. Data and models are versioned with DVC,
experiments are tracked with MLflow, and every push is checked by GitHub Actions.

Latest test-set result (credit task, 7,500 rows): **ROC-AUC 0.88**, PR-AUC 0.51, macro-F1 0.72.

## How it works

```
generate_data ─► create_dataset ─► tune ─► train ─► test ─► evaluate ─► register_model
 (synthetic      (time-based       (Optuna  (fit +   (predict  (metrics   (MLflow registry,
  CSVs)           train/val/test)   search)  thresh.) on test)  + plots)   "champion" alias)
```

| Stage | Script | Output |
|---|---|---|
| `generate_data` | `pipeline/generate_risklens_data.py` | `data/risklens/*.csv` |
| `create_dataset` | `pipeline/create_dataset.py` | `data/train.csv`, `val.csv`, `test.csv` |
| `tune` | `pipeline/tuning_experiment.py` | `best_params.json` |
| `train` | `pipeline/train.py` | `model.joblib`, `threshold.json` |
| `test` | `pipeline/test.py` | `data/predictions.csv` |
| `evaluate` | `pipeline/evaluate.py` | `metrics.json`, `plots/` |
| `register_model` | `pipeline/register_model.py` | `registry.json` |

Design choices worth knowing:
- **Time-based split.** The oldest rows train the model and the newest rows test it, so it is never evaluated on the past.
- **Tuned decision threshold.** Defaults are only 9% (credit) and 0.4% (fraud) of rows, so 0.5 is a poor cut-off. The threshold is chosen on the validation set to maximise macro-F1.
- **Five model families** are searched: logistic regression, random forest, histogram gradient boosting, XGBoost and LightGBM (see `mlcore/search_space.py`).
- **Champion model.** A new version only gets the `champion` alias if its test ROC-AUC beats the current champion.

## Project layout

```
mlcore/            reusable library: config, data, metrics, models, plots, search space, artifacts, MLflow helpers
pipeline/          one script per DVC stage (run as python -m pipeline.<name>)
orchestration/     Airflow and Prefect DAGs that run the same stages
tests/             pytest suite (unit tests + a small end-to-end test)
ci/                check_metrics.py - the model-quality gate used by CI
legacy/            the old pipeline, kept for reference, not maintained
dvc.yaml           the pipeline definition        params.yaml   all tunable settings
.github/workflows/ CI workflow                    docker-compose.yml   local S3 + MLflow + Airflow
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
```

**Run the whole pipeline** (starts from generated data, so nothing needs to be downloaded):
```bash
docker compose up -d mlflow       # MLflow UI at http://localhost:5000
dvc repro                         # runs only the stages whose inputs changed
```
Or on Windows: `run_mlflow.bat` (full) / `run_mlflow.bat quick` (skip tuning).

**Change a setting** in `params.yaml`, e.g. the task or the number of tuning trials, then run `dvc repro` again:

| Setting | Meaning |
|---|---|
| `dataset.task` | `credit` (probability of default) or `fraud` (card fraud) |
| `dataset.test_size` / `val_size` | share of the newest rows used for test / validation |
| `tuning.n_trials` | Optuna trials (40 by default) |
| `tuning.objective` | `roc_auc`, `macro_f1`, `accuracy` or `pr_auc` |
| `tuning.max_train_rows` | rows used while tuning (speed vs. accuracy) |

Compare two runs with `dvc params diff` and `dvc metrics diff`. More DVC details are in [guide_dvc.md](guide_dvc.md).

## Docker services

`docker compose up -d --build` starts:

| Service | URL | Purpose |
|---|---|---|
| `s3` | http://localhost:9090 | local S3 bucket used as the DVC remote |
| `mlflow` | http://localhost:5000 | experiment tracking and model registry |
| `airflow` | http://localhost:8080 | scheduled runs of the pipeline (login is printed by `docker logs airflow`) |

`mlcore/config.py` reads the MLflow address from the `MLFLOW_TRACKING_URI` environment variable.

## Tests and CI

```bash
ruff check .            # lint
pytest                  # 30 tests, about 10 seconds
pytest -m "not slow"    # skip the end-to-end test
```

The GitHub Actions workflow ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on every push and pull request:

1. **lint**: `ruff check`
2. **test**: `pytest`
3. **pipeline** (after 1 and 2 pass): `dvc exp run evaluate` with a reduced budget (4 trials, 5k tuning rows, about 1 minute), then `ci/check_metrics.py` fails the build if ROC-AUC drops below 0.75 or PR-AUC below 0.20. `metrics.json` and the plots are uploaded as the `model-report` artifact.

CI rebuilds the data from the seed in `params.yaml`, so it needs no DVC remote credentials. It skips `register_model`, which needs a running MLflow server.

## Tech stack

Python 3.13 · scikit-learn · XGBoost · LightGBM · Optuna · MLflow · DVC · Docker · Airflow / Prefect · pytest · Ruff · GitHub Actions
