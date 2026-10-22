# Register the trained model in the MLflow Model Registry.
#
# Flow: log run (params + metrics + model) -> register new version
#       -> if better than the current "champion", give it the "champion" alias.
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from mlcore.artifacts import load_pipeline, read_json, write_json
from mlcore.config import DATA_DIR, ID, METRICS_PATH, PLOTS_DIR, REGISTRY_PATH, TARGET, load_params
from mlcore.tracking import init_mlflow

# --- settings (easy to change) ---
EXPERIMENT = "classification"
MODEL_NAME = "risk_classifier"
ALIAS = "champion"  # the alias that means "the model to use"
METRIC = "roc_auc"  # higher is better


def log_and_register(model, metrics, params):
    """Log the run and register the model as a new version; returns the version number."""
    with mlflow.start_run():
        mlflow.log_params({f"{k}.{p}": v for k, d in params.items() for p, v in d.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifacts(str(PLOTS_DIR), "plots")
        sample = pd.read_csv(DATA_DIR / "test.csv").drop(columns=[TARGET, ID]).head(100)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",  # skops would need every pipeline type whitelisted
            registered_model_name=MODEL_NAME,
            signature=infer_signature(sample, model.predict_proba(sample)[:, 1]),
            input_example=sample.head(5),
        )
    return info.registered_model_version


def current_champion(client):
    """(model version, its test score), or (None, None) when no champion exists yet."""
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        return champion, client.get_run(champion.run_id).data.metrics[f"test_{METRIC}"]
    except MlflowException:
        return None, None


def promote_if_better(client, new_version, new_score):
    champion, champion_score = current_champion(client)
    if champion is None or new_score > champion_score:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, new_version)
        print(f"Version {new_version} is now '{ALIAS}'")
    else:
        print(
            f"Kept version {champion.version} as '{ALIAS}' "
            f"({METRIC} {champion_score:.4f} >= new {new_score:.4f})"
        )


def main():
    init_mlflow(EXPERIMENT)
    client = MlflowClient()

    model = load_pipeline()
    metrics = read_json(METRICS_PATH)

    new_version = log_and_register(model, metrics, load_params())
    print(f"Registered {MODEL_NAME} version {new_version} ({METRIC}={metrics[METRIC]:.4f})")

    promote_if_better(client, new_version, metrics[METRIC])

    # small file so DVC can track this step
    write_json(REGISTRY_PATH, {"model": MODEL_NAME, "version": int(new_version)})


if __name__ == "__main__":
    main()
