# backend/app/forecasting/execution.py

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from backend.app.forecasting.schemas import ForecastPlan, ForecastTarget, ForecastResult, ModelType

Frequency = Literal["daily", "weekly", "monthly", "quarterly", "yearly", "irregular", "unknown"]

def _ensure_datetime(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce", utc=False)
    out = out.dropna(subset=[dt_col])
    return out

def _infer_frequency(dt_index: pd.DatetimeIndex) -> str:
    try:
        f = pd.infer_freq(dt_index)
    except Exception:
        f = None

    if not f:
        return "unknown"

    f = f.upper()
    if f.startswith("D"):
        return "daily"
    if f.startswith("W"):
        return "weekly"
    if f.startswith("M"):
        return "monthly"
    if f.startswith("Q"):
        return "quarterly"
    if f.startswith("A") or f.startswith("Y"):
        return "yearly"
    return "irregular"


def _choose_auto_model(freq: str, n_points: int) -> str:
    if freq in ("daily", "weekly"):
        if n_points >= 120:
            return "prophet"
        return "arima"
    if freq in ("monthly", "quarterly", "yearly"):
        return "arima"
    if n_points >= 120:
        return "prophet"
    return "arima"


def _fit_predict_arima(y: pd.Series, horizon: int, seasonal: bool, seasonal_period: Optional[int],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as e:
        raise RuntimeError("statsmodels is required for ARIMA/SARIMA execution") from e

    y = y.astype(float)

    order = (1, 1, 1)

    if seasonal and seasonal_period and seasonal_period >= 2:
        seasonal_order = (1, 1, 1, int(seasonal_period))
    else:
        seasonal_order = (0, 0, 0, 0)

    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    pred = res.get_forecast(steps=horizon)
    yhat = pred.predicted_mean.to_numpy()
    ci = None
    try:
        ci = pred.conf_int(alpha=0.05)  # 95%
        lower = ci.iloc[:, 0].to_numpy()
        upper = ci.iloc[:, 1].to_numpy()
    except Exception:
        lower = upper = None

    meta = {
        "order": order,
        "seasonal_order": seasonal_order,
        "aic": getattr(res, "aic", None),
        "bic": getattr(res, "bic", None),
    }
    return yhat, lower, upper, meta


def _seasonal_period_from_frequency(freq: str) -> Optional[int]:
    if freq == "daily":
        return 7
    if freq == "weekly":
        return 52
    if freq == "monthly":
        return 12
    if freq == "quarterly":
        return 4
    return None


def _fit_predict_prophet( df: pd.DataFrame, dt_col: str, y_col: str, horizon: int, freq_label: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    Prophet = None
    import_err = None
    try:
        from prophet import Prophet
    except Exception as e1:
        import_err = e1
        try:
            from fbprophet import Prophet
        except Exception as e2:
            raise RuntimeError(
                "prophet (or fbprophet) is required for Prophet execution"
            ) from (import_err or e2)

    work = df[[dt_col, y_col]].copy()
    work = work.rename(columns={dt_col: "ds", y_col: "y"})
    work = work.dropna(subset=["ds", "y"])

    work["y"] = work["y"].astype(float)

    m = Prophet(
        yearly_seasonality=(freq_label in ("daily", "weekly", "monthly")),
        weekly_seasonality=(freq_label == "daily"),
        daily_seasonality=False,
    )
    m.fit(work)

    future_freq = "D"
    if freq_label == "weekly":
        future_freq = "W"
    elif freq_label == "monthly":
        future_freq = "MS"
    elif freq_label == "quarterly":
        future_freq = "QS"
    elif freq_label == "yearly":
        future_freq = "YS"

    future = m.make_future_dataframe(periods=horizon, freq=future_freq, include_history=False)
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    meta = {
        "prophet_freq": future_freq,
        "seasonality": {
            "yearly": bool(m.yearly_seasonality),
            "weekly": bool(m.weekly_seasonality),
            "daily": bool(m.daily_seasonality),
        },
    }
    return fcst, meta

def _make_lag_features(y: pd.Series, lags: int = 14) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame({"y": y.astype(float)})
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df = df.dropna()
    X = df.drop(columns=["y"])
    y_out = df["y"]
    return X, y_out


def _fit_predict_random_forest(y: pd.Series, horizon: int, lags: int = 14) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as e:
        raise RuntimeError("scikit-learn is required for RandomForest execution") from e

    y = y.astype(float)
    X, y_train = _make_lag_features(y, lags=lags)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y_train)

    history = list(y.dropna().astype(float).values)
    preds = []
    for _ in range(horizon):
        if len(history) < lags:
            preds.append(history[-1] if history else np.nan)
            history.append(preds[-1])
            continue
        x = np.array(history[-lags:][::-1], dtype=float)  #
        x = x.reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        history.append(yhat)

    meta = {"lags": lags, "n_estimators": 300}
    return np.array(preds, dtype=float), meta

def run_forecast(*, dataset_id: str, df: pd.DataFrame, plan: ForecastPlan, model: ModelType = "auto",
) -> List[ForecastResult]:
    """
    Deterministic execution:
    - uses plan (datetime_column, targets, group_by, mode)
    - runs chosen model per target (and per group if grouped)
    - returns list of results (one per target; if grouped, result df includes group column)
    """
    if plan.mode == "skipped" or not plan.suitable:
        return []

    if not plan.datetime_column:
        raise ValueError("ForecastPlan.datetime_column is required for execution")

    dt_col = plan.datetime_column
    work = _ensure_datetime(df, dt_col)

    work = work.sort_values(dt_col)
    freq = plan.inferred_frequency or _infer_frequency(pd.DatetimeIndex(work[dt_col]))
    if freq == "unknown":
        freq = "irregular"

    results: List[ForecastResult] = []

    group_by = plan.group_by if plan.mode == "grouped" else None
    if plan.mode == "grouped" and not group_by:
        group_by = None

    for tgt in plan.targets:
        y_col = tgt.column
        horizon = int(tgt.horizon)

        if y_col not in work.columns:
            continue

        if group_by and group_by in work.columns:
            frames = []
            meta_groups: Dict[str, Any] = {"groups": {}}

            for g_val, g_df in work[[dt_col, y_col, group_by]].groupby(group_by):
                g_df = g_df.dropna(subset=[y_col]).sort_values(dt_col)
                if len(g_df) < 10:
                    continue

                series = pd.Series(g_df[y_col].values, index=pd.DatetimeIndex(g_df[dt_col]))
                series = series.astype(float)

                chosen = model
                if chosen == "auto":
                    chosen = _choose_auto_model(freq, len(series))

                fcst_df, meta = _execute_one_series(
                    series=series,
                    dt_index=series.index,
                    horizon=horizon,
                    freq=freq,
                    chosen_model=chosen,
                )
                fcst_df[group_by] = g_val
                frames.append(fcst_df)
                meta_groups["groups"][str(g_val)] = meta

            if not frames:
                continue

            out_df = pd.concat(frames, ignore_index=True)

            results.append(
                ForecastResult(
                    dataset_id=dataset_id,
                    mode="grouped",
                    model_used=str(model),
                    datetime_column=dt_col,
                    target=y_col,
                    group_by=group_by,
                    horizon=horizon,
                    frequency=freq,
                    forecast_df=out_df,
                    meta=meta_groups,
                )
            )

        else:
            base = work[[dt_col, y_col]].dropna(subset=[y_col]).sort_values(dt_col)
            if len(base) < 10:
                continue

            series = pd.Series(base[y_col].values, index=pd.DatetimeIndex(base[dt_col])).astype(float)
            chosen = model
            if chosen == "auto":
                chosen = _choose_auto_model(freq, len(series))

            out_df, meta = _execute_one_series(
                series=series,
                dt_index=series.index,
                horizon=horizon,
                freq=freq,
                chosen_model=chosen,
            )

            results.append(
                ForecastResult(
                    dataset_id=dataset_id,
                    mode="overall",
                    model_used=str(chosen),
                    datetime_column=dt_col,
                    target=y_col,
                    group_by=None,
                    horizon=horizon,
                    frequency=freq,
                    forecast_df=out_df,
                    meta=meta,
                )
            )

    return results


def _execute_one_series(*, series: pd.Series, dt_index: pd.DatetimeIndex, horizon: int, freq: str, chosen_model: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    last_dt = dt_index.max()
    future_dt = _make_future_dates(last_dt, horizon, freq)

    if chosen_model == "prophet":
        tmp = pd.DataFrame({"dt": dt_index, "y": series.values})
        fcst, meta = _fit_predict_prophet(tmp, dt_col="dt", y_col="y", horizon=horizon, freq_label=freq)
        out = fcst.rename(columns={"ds": "dt"})[["dt", "yhat", "yhat_lower", "yhat_upper"]]
        return out, {"model": "prophet", **meta}

    if chosen_model == "random_forest":
        yhat, meta = _fit_predict_random_forest(series, horizon=horizon, lags=14)
        out = pd.DataFrame({"dt": future_dt, "yhat": yhat})
        return out, {"model": "random_forest", **meta}

    seasonal_period = _seasonal_period_from_frequency(freq)
    seasonal = seasonal_period is not None and len(series) >= (seasonal_period * 2)
    yhat, lower, upper, meta = _fit_predict_arima(
        y=series,
        horizon=horizon,
        seasonal=seasonal,
        seasonal_period=seasonal_period,
    )
    out = pd.DataFrame({"dt": future_dt, "yhat": yhat})
    if lower is not None and upper is not None:
        out["yhat_lower"] = lower
        out["yhat_upper"] = upper
    return out, {"model": "arima", "seasonal": seasonal, **meta}


def _make_future_dates(last_dt: pd.Timestamp, horizon: int, freq: str) -> List[pd.Timestamp]:

    if freq == "weekly":
        step = pd.offsets.Week(1)
    elif freq == "monthly":
        step = pd.offsets.MonthBegin(1)
    elif freq == "quarterly":
        step = pd.offsets.QuarterBegin(1)
    elif freq == "yearly":
        step = pd.offsets.YearBegin(1)
    else:
        step = pd.offsets.Day(1)

    dates = []
    cur = last_dt
    for _ in range(horizon):
        cur = cur + step
        dates.append(cur)
    return dates