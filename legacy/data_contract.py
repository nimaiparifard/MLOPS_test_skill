import numpy as np
import pandas as pd
import pandera.pandas as pa

TARGET = "target"
MIN_ROWS = 50


def build_schema(n_features: int) -> pa.DataFrameSchema:
    """Contract for a regression dataset: numeric, finite, non-null features and target."""
    finite = pa.Check(lambda s: np.isfinite(s), element_wise=False, error="values must be finite")

    columns = {
        f"x{i + 1}": pa.Column(float, checks=finite, nullable=False)
        for i in range(n_features)
    }
    columns[TARGET] = pa.Column(
        float,
        checks=[finite, pa.Check(lambda s: s.std() > 0, error="target must not be constant")],
        nullable=False,
    )

    return pa.DataFrameSchema(
        columns=columns,
        checks=pa.Check(lambda df: len(df) >= MIN_ROWS, error=f"need at least {MIN_ROWS} rows"),
        strict=True,  # no unexpected columns
        ordered=False,
    )


def validate_schema(df: pd.DataFrame, n_features: int) -> pd.DataFrame:
    """Validate a train/test dataframe; raises pandera.errors.SchemaErrors listing every failure."""
    return build_schema(n_features).validate(df, lazy=True)
