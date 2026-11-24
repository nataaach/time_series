import numpy as np
from statsmodels.stats.stattools import durbin_watson


class ModelEvaluator:
    """Обчислення ключових показників ефективності (KPI) моделей обробки та екстраполяції."""

    @staticmethod
    def regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train_last: float | None = None,
    ) -> dict:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        err = y_true - y_pred

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))

        denom = np.where(np.abs(y_true) > 1e-9, np.abs(y_true), 1e-9)
        mape = float(np.mean(np.abs(err) / denom) * 100.0)

        smape_denom = np.abs(y_true) + np.abs(y_pred) + 1e-9
        smape = float(np.mean(2.0 * np.abs(err) / smape_denom) * 100.0)

        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        if y_train_last is not None and len(y_true) > 0:
            naive_pred = np.empty_like(y_true)
            naive_pred[0] = y_train_last
            if len(y_true) > 1:
                naive_pred[1:] = y_true[:-1]
            naive_rmse = float(np.sqrt(np.mean((y_true - naive_pred) ** 2)))
            theil_u = float(rmse / (naive_rmse + 1e-12))
        else:
            theil_u = np.nan

        dw = float(durbin_watson(err)) if (len(err) >= 3 and ss_res > 1e-12) else np.nan

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_%": mape,
            "sMAPE_%": smape,
            "R2": r2,
            "Theil_U": theil_u,
            "Durbin_Watson": dw,
        }

    @staticmethod
    def compute_aic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        n = len(y_true)
        rss = max(float(np.sum((y_true - y_pred) ** 2)), 1e-12)
        return float(n * np.log(rss / n) + 2 * n_params)