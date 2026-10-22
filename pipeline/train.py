import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from mlcore.artifacts import read_json, save_model
from mlcore.config import BEST_PARAMS_PATH, MODEL_PATH, load_params
from mlcore.data import load_xy
from mlcore.metrics import best_threshold, compute_metrics, learning_curve
from mlcore.models import build_model
from mlcore.plots import confusion_figure, learning_curve_figure, pr_figure, roc_figure
from mlcore.tracking import init_mlflow, log_figure, log_splits


def log_learning_curve(lc, model_name):
    for r in lc.itertuples():  # step = rows, so MLflow also draws them in the Model metrics tab
        mlflow.log_metrics(
            {f"lc_{r.split}_accuracy": r.accuracy, f"lc_{r.split}_macro_f1": r.macro_f1, f"lc_{r.split}_roc_auc": r.roc_auc},
            step=r.n_train,
        )
    mlflow.log_table(lc, "learning_curve.json")
    log_figure(learning_curve_figure(lc, f"Learning curve - {model_name}"), "plots/learning_curve.png")


def log_fitted_model(model, X_train):
    sample = X_train.head(100)
    mlflow.sklearn.log_model(
        model,
        name="model",
        serialization_format="cloudpickle",  # skops would need every pipeline type whitelisted
        signature=infer_signature(sample, model.predict_proba(sample)[:, 1]),
        input_example=X_train.head(5),
    )


def main():
    init_mlflow("classification-experiment")
    cfg = load_params()
    task = cfg["dataset"]["task"]
    # model type + hyperparameters found by Optuna
    best = read_json(BEST_PARAMS_PATH)
    X_train, y_train = load_xy("train.csv")
    X_val, y_val = load_xy("val.csv")

    def make_model():
        return build_model(best["model"], best["params"], X_train)

    with mlflow.start_run(run_name=f"train-{task}-{best['model']}"):
        mlflow.set_tags({"task": task, "model": best["model"], "stage": "train"})
        mlflow.log_param("model", best["model"])
        mlflow.log_params(best["params"])
        mlflow.log_params({f"dataset.{k}": v for k, v in cfg["dataset"].items()})
        mlflow.log_params({"train_rows": len(X_train), "val_rows": len(X_val), "n_features": X_train.shape[1]})
        log_splits(train=(X_train, y_train), val=(X_val, y_val))

        model = make_model().fit(X_train, y_train)
        p_train = model.predict_proba(X_train)[:, 1]
        p_val = model.predict_proba(X_val)[:, 1]
        threshold = best_threshold(y_val, p_val)

        metrics = {f"train_{k}": v for k, v in compute_metrics(y_train, p_train, threshold).items()}
        metrics |= {f"val_{k}": v for k, v in compute_metrics(y_val, p_val, threshold).items()}
        metrics["threshold"] = threshold
        mlflow.log_metrics(metrics)

        curves = {"train": (y_train, p_train), "val": (y_val, p_val)}
        log_figure(roc_figure(curves), "plots/roc_curve.png")
        log_figure(pr_figure(curves), "plots/pr_curve.png")
        log_figure(confusion_figure(y_val, p_val, threshold, "Validation confusion matrix"), "plots/confusion_matrix_val.png")

        lc = learning_curve(
            make_model, X_train, y_train, X_val, y_val, cfg["train"]["learning_curve_fractions"], cfg["train"]["seed"]
        )
        log_learning_curve(lc, best["model"])
        log_fitted_model(model, X_train)
    return model, threshold


if __name__ == "__main__":
    model, threshold = main()
    save_model(model, threshold)
    print(f"Model saved to {MODEL_PATH} (decision threshold {threshold:.4f})")
