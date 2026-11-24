import time
import warnings
from typing import Optional
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.kpi_evaluator import ModelEvaluator


class ExponentialSmoothingOptimizer:
    """
    Група вимог 1: Автоматичний підбір оптимальної моделі експоненціального згладжування
    (SES, Holt Linear/Damped, Holt-Winters Additive/Multiplicative), навчання та екстраполяція.
    """

    @staticmethod
    def _build_candidates(train: np.ndarray, seasonal_period: Optional[int]) -> list[dict]:
        strictly_positive = bool(np.all(train > 0))
        candidates = [
            {"name": "SES", "trend": None, "damped_trend": False, "seasonal": None, "seasonal_periods": None},
            {"name": "Holt_Add", "trend": "add", "damped_trend": False, "seasonal": None, "seasonal_periods": None},
            {"name": "Holt_Add_Damped", "trend": "add", "damped_trend": True, "seasonal": None, "seasonal_periods": None},
        ]
        if strictly_positive:
            candidates.append(
                {"name": "Holt_Mul", "trend": "mul", "damped_trend": False, "seasonal": None, "seasonal_periods": None}
            )

        if seasonal_period is not None and seasonal_period >= 2 and len(train) >= 2 * seasonal_period:
            candidates.extend([
                {
                    "name": "HW_Add_Add",
                    "trend": "add",
                    "damped_trend": False,
                    "seasonal": "add",
                    "seasonal_periods": seasonal_period,
                },
                {
                    "name": "HW_Add_Damped_Add",
                    "trend": "add",
                    "damped_trend": True,
                    "seasonal": "add",
                    "seasonal_periods": seasonal_period,
                },
            ])
            if strictly_positive:
                candidates.extend([
                    {
                        "name": "HW_Add_Mul",
                        "trend": "add",
                        "damped_trend": False,
                        "seasonal": "mul",
                        "seasonal_periods": seasonal_period,
                    },
                    {
                        "name": "HW_Mul_Mul",
                        "trend": "mul",
                        "damped_trend": False,
                        "seasonal": "mul",
                        "seasonal_periods": seasonal_period,
                    },
                ])
        return candidates

    @staticmethod
    def _fit_single(series: np.ndarray, cfg: dict):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                series,
                trend=cfg["trend"],
                damped_trend=cfg["damped_trend"],
                seasonal=cfg["seasonal"],
                seasonal_periods=cfg["seasonal_periods"],
                initialization_method="estimated",
            )
            return model.fit(optimized=True)

    def fit_and_forecast(
        self,
        train: np.ndarray,
        test_horizon: int,
        future_horizon: int,
        seasonal_period: Optional[int] = None,
    ) -> dict:
        t0 = time.perf_counter()
        train = np.asarray(train, dtype=float)
        candidates = self._build_candidates(train, seasonal_period)

        val_size = max(4, min(test_horizon, len(train) // 5))
        use_val = (len(train) - val_size) >= max(10, 2 * (seasonal_period or 2))

        leaderboard = []
        best_cfg = None
        best_score = np.inf

        for cfg in candidates:
            try:
                if use_val:
                    sub_train, sub_val = train[:-val_size], train[-val_size:]
                    res_sub = self._fit_single(sub_train, cfg)
                    val_pred = np.asarray(res_sub.forecast(val_size), dtype=float)
                    val_rmse = float(np.sqrt(np.mean((sub_val - val_pred) ** 2)))
                    score = val_rmse
                else:
                    res_full = self._fit_single(train, cfg)
                    score = float(res_full.aic)
                    val_rmse = np.nan

                res_tmp = self._fit_single(train, cfg)
                leaderboard.append({
                    "model": cfg["name"],
                    "val_RMSE": val_rmse,
                    "AIC": float(res_tmp.aic),
                    "BIC": float(res_tmp.bic),
                })

                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_cfg = cfg
            except Exception:
                continue

        if best_cfg is None:
            best_cfg = candidates[0]

        best_res = self._fit_single(train, best_cfg)
        fitted_train = np.asarray(best_res.fittedvalues, dtype=float)
        total_forecast = np.asarray(best_res.forecast(test_horizon + future_horizon), dtype=float)
        test_pred = total_forecast[:test_horizon]
        future_pred = total_forecast[test_horizon:]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        train_kpi = ModelEvaluator.regression_metrics(train, fitted_train)
        params_dict = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in best_res.params.items()
            if k in ("smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend")
        }

        return {
            "paradigm": "ExponentialSmoothing",
            "best_model_name": best_cfg["name"],
            "params": params_dict,
            "aic": float(best_res.aic),
            "bic": float(best_res.bic),
            "fit_time_ms": float(elapsed_ms),
            "fitted_train": fitted_train,
            "test_pred": test_pred,
            "future_pred": future_pred,
            "train_kpi": train_kpi,
            "leaderboard": sorted(
                leaderboard,
                key=lambda x: x["val_RMSE"] if np.isfinite(x["val_RMSE"]) else x["AIC"],
            ),
        }