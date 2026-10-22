from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
}


def build_model(name: str, params: dict, X):
    """Preprocessing + classifier in one Pipeline; used by tuning and training.

    X only tells us which columns are numeric (median-imputed, scaled) and which are
    categorical (one-hot), so the same pipeline works for the credit and the fraud data.
    """
    numeric = X.select_dtypes("number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    return Pipeline([("prep", prep), ("clf", MODELS[name](**params))])
