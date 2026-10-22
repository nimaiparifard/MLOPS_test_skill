"""MLflow helpers shared by the tuning, training and registration stages."""
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from .config import MLFLOW_URI, TARGET


def init_mlflow(experiment):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment)


def log_figure(fig, path):
    """Log a matplotlib figure to the active run and free it."""
    mlflow.log_figure(fig, path)
    plt.close(fig)


def log_splits(**splits):
    """Log datasets as run inputs: log_splits(train=(X, y), val=(X, y))."""
    for name, (X, y) in splits.items():
        frame = pd.concat([X, y], axis=1)
        mlflow.log_input(mlflow.data.from_pandas(frame, name=name, targets=TARGET), context=name)
