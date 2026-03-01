from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def safe_mape(y_true, y_pred, eps: float = 1e-9) -> float:
    """Mean Absolute Percentage Error (MAPE) in % with safe divide."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def mean_bias_error(y_true, y_pred) -> float:
    """Mean Bias Error (MBE): positive means over-prediction on average."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def regression_metrics(y_true, y_pred) -> dict:
    """Compute common regression metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    mbe = mean_bias_error(y_true, y_pred)
    return {
        "R-squared (R²)": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "MBE (Bias)": mbe,
    }


def validate_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce a series to numeric, dropping non-numeric rows later."""
    return pd.to_numeric(s, errors="coerce")