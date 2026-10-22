"""Shared fixtures: a small synthetic table that looks like the real train / val files."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def frame():
    """200 rows: two numeric features (one with NaNs), one categorical, ~20% positives."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "income": x,
            "age": rng.integers(18, 80, size=n).astype(float),
            "purpose": rng.choice(["car", "home", "education"], size=n),
        }
    )
    df.loc[::17, "age"] = np.nan  # missing values must be imputed, not crash the pipeline
    # the target depends on `income`, so a model can learn something
    df["target"] = (x + rng.normal(scale=0.5, size=n) > 0.8).astype(int)
    return df
