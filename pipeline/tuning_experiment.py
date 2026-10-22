# Tune hyperparameters with Optuna and track everything in MLflow.
#
# Each Optuna trial picks one of 5 classifiers + its hyperparameters, fits on train.csv
# (stratified subsample, see tuning.max_train_rows) and is scored on val.csv. The best choice
# is saved to best_params.json, which train.py then uses.
#
# In MLflow you get (experiment "tuning-experiment"):
#   parent run "optuna-study": search space, best result, per-model bests, trials table, and
#     plots/optuna/*.html            Optuna history, slice, parallel coordinate, importances
#     plots/model_comparison.png     accuracy / macro F1 / ROC AUC of each model's best trial
#     plots/roc_by_model.png         validation ROC curve of each model's best trial
#     plots/sensitivity/<model>/*.png  line plots: accuracy, macro F1 and ROC AUC while ONE
#                                      hyperparameter moves and the others stay at their best value
#   nested run per trial: model, hyperparameters, train/val metrics, fit time
import time

import mlflow
import optuna
import optuna.visualization as vis
import pandas as pd

from mlcore.artifacts import write_json
from mlcore.config import BEST_PARAMS_PATH, load_params
from mlcore.data import load_xy, stratified_subsample
from mlcore.metrics import SCORES, best_threshold, compute_metrics
from mlcore.models import build_model
from mlcore.plots import comparison_figure, roc_figure, sweep_figure
from mlcore.search_space import SEARCH_SPACE, describe_range, suggest, sweep_values
from mlcore.tracking import init_mlflow, log_figure, log_splits


def evaluate(name, params, data):
    """Fit on the (sub-sampled) train set and score on validation at the macro-F1-optimal cut-off."""
    X_train, y_train, X_val, y_val = data
    start = time.perf_counter()
    model = build_model(name, params, X_train).fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    p_val = model.predict_proba(X_val)[:, 1]
    threshold = best_threshold(y_val, p_val)
    val = compute_metrics(y_val, p_val, threshold)
    train = compute_metrics(y_train, model.predict_proba(X_train)[:, 1], threshold)
    return val, train, threshold, fit_seconds, p_val


def run_sensitivity(name, best_params, n_points, data):
    """One-at-a-time sweep around the best trial of a model -> one line plot per hyperparameter."""
    rows = []
    for p, spec in SEARCH_SPACE[name].items():
        if spec["kind"] == "fixed":
            continue
        for value in sweep_values(spec, n_points):
            val, *_ = evaluate(name, {**best_params, p: value}, data)
            rows.append({"param": p, "value": value, **{s: val[s] for s in SCORES}})
        df = pd.DataFrame(rows)
        df = df[df["param"] == p]
        log_figure(
            sweep_figure(df, name, p, log_x=spec.get("log", False), best_value=best_params[p]),
            f"plots/sensitivity/{name}/{p}.png",
        )
    return pd.DataFrame(rows).assign(model=name)


def load_tuning_data(cfg):
    """Train (stratified-subsampled to tuning.max_train_rows) and validation sets."""
    X_train, y_train = load_xy("train.csv")
    X_val, y_val = load_xy("val.csv")
    if len(X_train) > cfg["max_train_rows"]:
        X_train, y_train = stratified_subsample(X_train, y_train, cfg["max_train_rows"], cfg["seed"])
    return X_train, y_train, X_val, y_val


def make_objective(objective_name, data, best_per_model):
    """Optuna objective: one nested MLflow run per trial; fills `best_per_model` as a side effect."""

    def objective(trial):
        name, params = suggest(trial)
        with mlflow.start_run(run_name=f"trial-{trial.number:02d}-{name}", nested=True):
            mlflow.set_tags({"trial_number": trial.number, "model": name})
            mlflow.log_param("model", name)
            mlflow.log_params(params)

            val, train, threshold, fit_seconds, p_val = evaluate(name, params, data)
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in val.items()}
                | {f"train_{k}": v for k, v in train.items()}
                | {"threshold": threshold, "fit_seconds": fit_seconds}
            )
        value = val[objective_name]
        trial.set_user_attr("model_name", name)
        trial.set_user_attr("model_params", params)
        trial.set_user_attr("metrics", val)
        if name not in best_per_model or value > best_per_model[name]["value"]:
            best_per_model[name] = {"value": value, "params": params, "metrics": val, "p_val": p_val}
        return value

    return objective


def log_study_setup(cfg, data):
    X_train, y_train, X_val, y_val = data
    objective_name = cfg["objective"]
    mlflow.set_tags(
        {
            "mlflow.note.content": (
                f"Optuna search over {len(SEARCH_SPACE)} classifiers. Objective: maximize {objective_name} "
                "on val.csv. Accuracy / macro F1 use the macro-F1-optimal cut-off. Artifacts: "
                "search_space.json, trials.json, per_model_summary.json, sensitivity.json and plots/."
            ),
            "objective": f"maximize val_{objective_name}",
            "sampler": "TPE",
        }
    )
    mlflow.log_params(
        {
            "n_trials": cfg["n_trials"],
            "seed": cfg["seed"],
            "objective": objective_name,
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "models": ",".join(SEARCH_SPACE),
        }
    )
    log_splits(train=(X_train, y_train), val=(X_val, y_val))

    # search space: a file with everything + one readable param per range
    mlflow.log_dict(SEARCH_SPACE, "search_space.json")
    for model_name, space in SEARCH_SPACE.items():
        for p, spec in space.items():
            mlflow.log_param(f"space.{model_name}.{p}", describe_range(spec))


def log_trial_curve(study, objective_name):
    """Metric over trials -> line chart in the run's "Model metrics" tab."""
    best_so_far = -float("inf")
    for t in study.trials:
        best_so_far = max(best_so_far, t.value)
        mlflow.log_metrics(
            {f"trial_val_{objective_name}": t.value, f"best_val_{objective_name}_so_far": best_so_far},
            step=t.number,
        )


def summarize_best(study, objective_name):
    best = {
        "model": study.best_trial.user_attrs["model_name"],
        "params": study.best_trial.user_attrs["model_params"],
        "objective": objective_name,
        "value": study.best_value,
        "val_metrics": study.best_trial.user_attrs["metrics"],
    }
    mlflow.log_metric(f"best_val_{objective_name}", study.best_value)
    mlflow.log_metrics({f"best_val_{k}": v for k, v in best["val_metrics"].items()})
    mlflow.log_param("best_model", best["model"])
    mlflow.log_params({f"best.{k}": v for k, v in best["params"].items()})
    return best


def log_trial_tables(study, objective_name):
    """Best per model + full trial table."""
    trials = study.trials_dataframe()
    trials["model"] = [t.user_attrs["model_name"] for t in study.trials]
    per_model = trials.groupby("model")["value"].agg(["count", "max", "mean"])
    for model_name, row in per_model.iterrows():
        mlflow.log_metrics(
            {f"best_val_{objective_name}.{model_name}": row["max"], f"trials.{model_name}": row["count"]}
        )
    mlflow.log_table(per_model.reset_index(), "per_model_summary.json")
    mlflow.log_table(trials, "trials.json")
    mlflow.log_text(trials.to_csv(index=False), "trials.csv")


def log_optuna_plots(study):
    """Interactive Optuna plots (open the .html files in the run's Artifacts tab)."""
    optuna_plots = {
        "optimization_history": vis.plot_optimization_history,
        "hyperparameter_vs_metric": vis.plot_slice,
        "parallel_coordinate": vis.plot_parallel_coordinate,
        "param_importances": vis.plot_param_importances,
    }
    for name, make_plot in optuna_plots.items():
        try:
            mlflow.log_figure(make_plot(study), f"plots/optuna/{name}.html")
        except Exception as err:  # e.g. importances need enough varied trials
            print(f"skipped plot {name}: {err}")


def log_model_comparison(best_per_model, y_val):
    """Model comparison + ROC curves of every model's best trial."""
    summary = pd.DataFrame({m: b["metrics"] for m, b in best_per_model.items()}).T
    log_figure(comparison_figure(summary), "plots/model_comparison.png")
    log_figure(
        roc_figure({m: (y_val, b["p_val"]) for m, b in best_per_model.items()}, "Validation ROC, best trial per model"),
        "plots/roc_by_model.png",
    )


def log_sensitivity(best_per_model, n_points, data):
    """Hyperparameter line plots (accuracy / macro F1 / ROC AUC) for every model."""
    sweeps = [run_sensitivity(m, b["params"], n_points, data) for m, b in best_per_model.items()]
    table = pd.concat(sweeps, ignore_index=True)
    mlflow.log_table(table.assign(value=table["value"].astype(str)), "sensitivity.json")


def main():
    init_mlflow("tuning-experiment")
    cfg = load_params("tuning")
    objective_name = cfg["objective"]
    data = load_tuning_data(cfg)
    y_val = data[3]

    best_per_model = {}  # name -> dict(value, params, val metrics, p_val)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="classification-tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
    )

    with mlflow.start_run(run_name="optuna-study"):
        log_study_setup(cfg, data)
        study.optimize(make_objective(objective_name, data, best_per_model), n_trials=cfg["n_trials"])
        log_trial_curve(study, objective_name)
        best = summarize_best(study, objective_name)
        log_trial_tables(study, objective_name)
        log_optuna_plots(study)
        log_model_comparison(best_per_model, y_val)
        if cfg["sweep_points"] > 1:
            log_sensitivity(best_per_model, cfg["sweep_points"], data)

    write_json(BEST_PARAMS_PATH, best)
    print(f"Best: {best['model']} {best['params']}  val_{objective_name}={best['value']:.4f}")


if __name__ == "__main__":
    main()
