import time
import warnings
from typing import Optional
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.kpi_evaluator import ModelEvaluator


class RegressionOptimizer:
    @staticmethod
    def _build_time_fourier_features(
        t_indices: np.ndarray,
        degree: int,
        seasonal_period: Optional[int],
    ) -> np.ndarray:
        cols = []
        for d in range(1, degree + 1):
            cols.append(t_indices ** d)

        if seasonal_period is not None and seasonal_period >= 2:
            if seasonal_period == 2:
                cols.append(np.cos(np.pi * t_indices))
            else:
                n_harmonics = min(3, (seasonal_period - 1) // 2)
                for k in range(1, n_harmonics + 1):
                    angle = 2.0 * np.pi * k * t_indices / seasonal_period
                    cols.append(np.sin(angle))
                    cols.append(np.cos(angle))

        return np.column_stack(cols)

    def _fit_predict_static(
        self,
        train: np.ndarray,
        t_start: int,
        horizon: int,
        degree: int,
        seasonal_period: Optional[int],
        estimator_type: str,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        n = len(train)
        t_train = np.arange(t_start, t_start + n, dtype=float)
        t_pred = np.arange(t_start + n, t_start + n + horizon, dtype=float)

        X_train = self._build_time_fourier_features(t_train, degree, seasonal_period)
        X_pred = self._build_time_fourier_features(t_pred, degree, seasonal_period)

        if estimator_type == "ols":
            pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        elif estimator_type == "ridge":
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        elif estimator_type == "huber":
            pipe = Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(max_iter=500, epsilon=1.35))])
        else:
            raise ValueError(f"Невідомий тип моделі: {estimator_type}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, train)

        fitted = pipe.predict(X_train)
        forecast = pipe.predict(X_pred)
        n_params = X_train.shape[1] + 1
        return fitted, forecast, n_params

    def _fit_predict_dynamic_arx(
        self,
        train: np.ndarray,
        t_start: int,
        horizon: int,
        lags: int,
        seasonal_period: Optional[int],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        n = len(train)
        lags = min(lags, max(1, n // 4))
        t_all = np.arange(t_start, t_start + n, dtype=float)
        exog_train = self._build_time_fourier_features(t_all, degree=1, seasonal_period=seasonal_period)

        X_rows, y_rows = [], []
        for idx in range(lags, n):
            lag_feats = [train[idx - k] for k in range(1, lags + 1)]
            X_rows.append(np.concatenate([lag_feats, exog_train[idx]]))
            y_rows.append(train[idx])

        X_mat = np.asarray(X_rows, dtype=float)
        y_vec = np.asarray(y_rows, dtype=float)

        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=2.0))])
        pipe.fit(X_mat, y_vec)

        fitted = np.empty(n, dtype=float)
        fitted[:lags] = train[:lags]
        fitted[lags:] = pipe.predict(X_mat)

        history = list(train)
        forecasts = []
        for step in range(horizon):
            t_curr = np.array([float(t_start + n + step)])
            exog_curr = self._build_time_fourier_features(t_curr, degree=1, seasonal_period=seasonal_period)[0]
            lag_feats = [history[-k] for k in range(1, lags + 1)]
            x_step = np.concatenate([lag_feats, exog_curr]).reshape(1, -1)
            y_hat = float(pipe.predict(x_step)[0])
            forecasts.append(y_hat)
            history.append(y_hat)

        n_params = X_mat.shape[1] + 1
        return fitted, np.asarray(forecasts, dtype=float), n_params

    def fit_and_forecast(
        self,
        train: np.ndarray,
        test_horizon: int,
        future_horizon: int,
        seasonal_period: Optional[int] = None,
    ) -> dict:
        t0 = time.perf_counter()
        train = np.asarray(train, dtype=float)
        n = len(train)

        candidates = [
            {"name": "LinearTrend_Fourier_OLS", "kind": "static", "degree": 1, "est": "ols"},
            {"name": "Poly2_Fourier_Ridge", "kind": "static", "degree": 2, "est": "ridge"},
            {"name": "Poly3_Fourier_Ridge", "kind": "static", "degree": 3, "est": "ridge"},
            {"name": "Huber_Robust_Fourier", "kind": "static", "degree": 2, "est": "huber"},
            {"name": "Dynamic_ARX_Ridge", "kind": "arx", "lags": min(5, max(2, (seasonal_period or 4) // 2))},
        ]

        val_size = max(4, min(test_horizon, n // 5))
        use_val = (n - val_size) >= 12

        leaderboard = []
        best_cfg = None
        best_score = np.inf

        for cfg in candidates:
            try:
                if use_val:
                    sub_train, sub_val = train[:-val_size], train[-val_size:]
                    if cfg["kind"] == "static":
                        _, val_pred, _ = self._fit_predict_static(
                            sub_train, 0, val_size, cfg["degree"], seasonal_period, cfg["est"]
                        )
                    else:
                        _, val_pred, _ = self._fit_predict_dynamic_arx(
                            sub_train, 0, val_size, cfg["lags"], seasonal_period
                        )
                    val_rmse = float(np.sqrt(np.mean((sub_val - val_pred) ** 2)))
                    score = val_rmse
                else:
                    val_rmse = np.nan

                if cfg["kind"] == "static":
                    fit_full, _, k_params = self._fit_predict_static(
                        train, 0, 1, cfg["degree"], seasonal_period, cfg["est"]
                    )
                else:
                    fit_full, _, k_params = self._fit_predict_dynamic_arx(
                        train, 0, 1, cfg["lags"], seasonal_period
                    )
                aic = ModelEvaluator.compute_aic(train, fit_full, k_params)
                if not use_val:
                    score = aic

                leaderboard.append({
                    "model": cfg["name"],
                    "val_RMSE": val_rmse,
                    "AIC": aic,
                    "n_params": k_params,
                })

                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_cfg = cfg
            except Exception:
                continue

        if best_cfg is None:
            best_cfg = candidates[0]

        total_h = test_horizon + future_horizon
        if best_cfg["kind"] == "static":
            fitted_train, total_forecast, n_params = self._fit_predict_static(
                train, 0, total_h, best_cfg["degree"], seasonal_period, best_cfg["est"]
            )
        else:
            fitted_train, total_forecast, n_params = self._fit_predict_dynamic_arx(
                train, 0, total_h, best_cfg["lags"], seasonal_period
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        aic = ModelEvaluator.compute_aic(train, fitted_train, n_params)
        train_kpi = ModelEvaluator.regression_metrics(train, fitted_train)

        return {
            "paradigm": "Regression",
            "best_model_name": best_cfg["name"],
            "n_params": n_params,
            "aic": float(aic),
            "fit_time_ms": float(elapsed_ms),
            "fitted_train": fitted_train,
            "test_pred": total_forecast[:test_horizon],
            "future_pred": total_forecast[test_horizon:],
            "train_kpi": train_kpi,
            "leaderboard": sorted(
                leaderboard,
                key=lambda x: x["val_RMSE"] if np.isfinite(x["val_RMSE"]) else x["AIC"],
            ),
        }