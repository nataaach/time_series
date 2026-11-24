import os
import time
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True


@dataclass
class ModelMetrics:
    model_name: str
    category: str
    params: str
    mae: float
    rmse: float
    mape: float
    r2: float
    aic: Optional[float]
    bic: Optional[float]
    ljung_box_p: Optional[float]
    exec_time_sec: float


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    category: str,
    params: str,
    exec_time: float,
    aic: Optional[float] = None,
    bic: Optional[float] = None,
) -> ModelMetrics:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = compute_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))

    lb_p = None
    if len(residuals) > 12 and float(np.std(residuals)) > 1e-12:
        try:
            lag = max(1, min(10, len(residuals) // 3))
            lb_res = acorr_ljungbox(residuals, lags=[lag], return_df=True)
            lb_p = float(lb_res["lb_pvalue"].iloc[0])
        except Exception:
            lb_p = None

    return ModelMetrics(
        model_name=model_name,
        category=category,
        params=params,
        mae=mae,
        rmse=rmse,
        mape=mape,
        r2=r2,
        aic=aic,
        bic=bic,
        ljung_box_p=lb_p,
        exec_time_sec=exec_time,
    )


class TimeSeriesLoader:
    @staticmethod
    def load_data(
        file_path: str,
        date_col: Optional[str] = "Timestamp",
        value_col: Optional[str] = "Value",
    ) -> pd.Series:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не знайдено: {path}")

        df = pd.read_csv(path, sep=None, engine="python")
        print(f"[INFO] Завантажено файл: {path}")
        print(f"[INFO] Розмір таблиці: {df.shape[0]} рядків, {df.shape[1]} стовпців")
        print(f"[INFO] Стовпці: {list(df.columns)}")
        print("[INFO] Перші 3 рядки файлу:")
        print(df.head(3).to_string(index=False))

        if date_col not in df.columns:
            date_col = df.columns[0]
        if value_col not in df.columns:
            value_col = df.columns[-1]

        vals = df[value_col]
        if vals.dtype == object:
            vals = vals.astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        df[value_col] = pd.to_numeric(vals, errors="coerce")

        parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
        if parsed_dates.isna().mean() > 0.5:
            parsed_dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

        df[date_col] = parsed_dates
        df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
        series = pd.Series(df[value_col].to_numpy(dtype=float), index=df[date_col], name=str(value_col))
        if series.index.has_duplicates:
            series = series.groupby(level=0).mean()

        print(f"[INFO] Обрано дату='{date_col}', значення='{value_col}', N={len(series)}")
        return series


class TimeSeriesAnalyzer:
    def __init__(self, series: pd.Series, output_dir: str):
        self.series = series
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _safe_adf(arr: pd.Series) -> Tuple[float, float]:
        arr = arr.dropna()
        if len(arr) < 5 or float(arr.std()) < 1e-12:
            return 0.0, 1.0
        stat, p, *_ = adfuller(arr, autolag="AIC")
        return float(stat), float(p)

    @staticmethod
    def _safe_kpss(arr: pd.Series) -> Tuple[float, float]:
        arr = arr.dropna()
        if len(arr) < 5 or float(arr.std()) < 1e-12:
            return 0.0, 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, *_ = kpss(arr, regression="ct", nlags="auto")
        return float(stat), float(p)

    def analyze_properties(self, seasonal_period: int = 30) -> Dict:
        s = self.series
        desc = s.describe()

        adf_stat, adf_p = self._safe_adf(s)
        diff1 = s.diff().dropna()
        adf_d1_stat, adf_d1_p = self._safe_adf(diff1)
        kpss_stat, kpss_p = self._safe_kpss(s)

        results = {
            "N": len(s),
            "mean": float(desc["mean"]),
            "std": float(desc["std"]),
            "min": float(desc["min"]),
            "max": float(desc["max"]),
            "skewness": float(stats.skew(s)),
            "kurtosis": float(stats.kurtosis(s)),
            "adf_stat": adf_stat,
            "adf_p": adf_p,
            "adf_stationary": adf_p < 0.05,
            "adf_diff1_stat": adf_d1_stat,
            "adf_diff1_p": adf_d1_p,
            "adf_diff1_stationary": adf_d1_p < 0.05,
            "kpss_stat": kpss_stat,
            "kpss_p": kpss_p,
            "kpss_stationary": kpss_p >= 0.05,
        }

        self._plot_diagnostics(seasonal_period)
        return results

    def _plot_diagnostics(self, seasonal_period: int):
        if len(self.series) >= 2 * seasonal_period:
            decomp = seasonal_decompose(self.series, model="additive", period=seasonal_period)
            fig = decomp.plot()
            fig.set_size_inches(12, 9)
            fig.suptitle("Декомпозиція часового ряду (Тренд, Сезонність, Залишки)", fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "01_decomposition.png"), dpi=150)
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        diff1 = self.series.diff().dropna()
        max_lags = max(2, min(40, len(diff1) // 2 - 1))
        plot_acf(self.series, lags=max_lags, ax=axes[0, 0], title="ACF вихідного ряду")
        plot_pacf(self.series, lags=max_lags, ax=axes[0, 1], title="PACF вихідного ряду")
        if float(diff1.std()) > 1e-12:
            plot_acf(diff1, lags=max_lags, ax=axes[1, 0], title="ACF перших різниць (d=1)")
            plot_pacf(diff1, lags=max_lags, ax=axes[1, 1], title="PACF перших різниць (d=1)")
        else:
            axes[1, 0].plot(diff1.values)
            axes[1, 0].set_title("Перші різниці (константа)")
            axes[1, 1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "02_acf_pacf.png"), dpi=150)
        plt.close(fig)


class MovingAverageModels:
    @staticmethod
    def forecast_recursive(
        history: np.ndarray,
        steps: int,
        method: str,
        window: int,
        trend_damping: bool = False,
    ) -> np.ndarray:
        buf = list(history.astype(float))
        preds = []
        eff_w = min(window, len(buf))

        if trend_damping and eff_w >= 2:
            y_win = np.array(buf[-eff_w:])
            x_win = np.arange(eff_w)
            slope = float(np.polyfit(x_win, y_win, 1)[0])
        else:
            slope = 0.0

        alpha = 2.0 / (eff_w + 1.0)
        last_ema = float(pd.Series(history).ewm(span=eff_w, adjust=False).mean().iloc[-1])

        for step in range(steps):
            if method == "SMA":
                base = float(np.mean(buf[-eff_w:]))
            elif method == "WMA":
                w = np.arange(1, eff_w + 1, dtype=float)
                w /= w.sum()
                base = float(np.dot(buf[-eff_w:], w))
            elif method == "EMA":
                base = last_ema
            else:
                raise ValueError(f"Невідомий метод: {method}")

            damped_drift = slope * (0.95 ** step) if trend_damping else 0.0
            val = base + damped_drift
            preds.append(val)
            buf.append(val)
            if method == "EMA":
                last_ema = alpha * val + (1.0 - alpha) * last_ema

        return np.array(preds)

    @classmethod
    def optimize_window_rd(
        cls,
        train: np.ndarray,
        val: np.ndarray,
        method: str,
        window_grid: List[int],
        trend_damping: bool = False,
    ) -> Tuple[int, float, List[Tuple[int, float]]]:
        history_results = []
        valid_windows = [w for w in window_grid if w < len(train)]
        if not valid_windows:
            valid_windows = [max(2, len(train) // 3)]

        best_w = valid_windows[0]
        best_rmse = float("inf")

        for w in valid_windows:
            preds = []
            hist = list(train)
            for actual in val:
                pred = cls.forecast_recursive(np.array(hist), 1, method, w, trend_damping=trend_damping)[0]
                preds.append(pred)
                hist.append(actual)
            rmse = float(np.sqrt(mean_squared_error(val, preds)))
            history_results.append((w, rmse))
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = w

        return best_w, best_rmse, history_results


class AutoRegressiveModels:
    @staticmethod
    def fit_and_predict_arima(
        train: np.ndarray,
        steps: int,
        order: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, float, float, object]:
        fitted = ARIMA(train, order=order).fit()
        forecast = fitted.forecast(steps=steps)
        return np.asarray(forecast), float(fitted.aic), float(fitted.bic), fitted

    @staticmethod
    def grid_search_arima_rd(
        train: np.ndarray,
        val: np.ndarray,
        p_range: range,
        d_range: range,
        q_range: range,
    ) -> Tuple[Tuple[int, int, int], pd.DataFrame]:
        records = []
        best_order = (1, 1, 1)
        best_score = float("inf")

        for d in d_range:
            for p in p_range:
                for q in q_range:
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = ARIMA(train, order=(p, d, q)).fit()
                        preds = model.forecast(steps=len(val))
                        val_rmse = float(np.sqrt(mean_squared_error(val, preds)))
                        records.append({
                            "p": p, "d": d, "q": q,
                            "AIC": float(model.aic),
                            "BIC": float(model.bic),
                            "Val_RMSE": val_rmse,
                        })
                        if val_rmse < best_score:
                            best_score = val_rmse
                            best_order = (p, d, q)
                    except Exception:
                        continue

        df_grid = pd.DataFrame(records).sort_values("Val_RMSE") if records else pd.DataFrame()
        return best_order, df_grid


class StatisticalLearningModels:
    @staticmethod
    def fit_predict_holtwinters(
        train: np.ndarray,
        steps: int,
        seasonal_period: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, float]:
        if seasonal_period and len(train) >= 2 * seasonal_period:
            model = ExponentialSmoothing(
                train, trend="add", seasonal="add", seasonal_periods=seasonal_period, damped_trend=True
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(train, trend="add", damped_trend=True).fit(optimized=True)
        preds = model.forecast(steps)
        return np.asarray(preds), float(model.aic), float(model.bic)

    @staticmethod
    def fit_predict_ridge_lags(
        train: np.ndarray,
        steps: int,
        n_lags: int = 14,
        alpha: float = 10.0,
    ) -> np.ndarray:
        n = len(train)
        eff_lags = min(n_lags, max(2, n // 4))
        X, y = [], []
        for i in range(eff_lags, n):
            lags = train[i - eff_lags:i]
            time_feat = [i / n, (i / n) ** 2]
            X.append(np.concatenate([lags, time_feat]))
            y.append(train[i])

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        pipe.fit(np.array(X), np.array(y))

        buf = list(train.astype(float))
        preds = []
        for s in range(steps):
            idx = n + s
            lags = np.array(buf[-eff_lags:])
            time_feat = np.array([idx / n, (idx / n) ** 2])
            x_next = np.concatenate([lags, time_feat]).reshape(1, -1)
            y_next = float(pipe.predict(x_next)[0])
            preds.append(y_next)
            buf.append(y_next)

        return np.array(preds)


class TimeSeriesPipeline:
    def __init__(self, series: pd.Series, output_dir: str):
        self.series = series
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.N = len(series)
        self.seasonal_period = 30 if self.N >= 90 else (7 if self.N >= 28 else 4)

        n_train = int(self.N * 0.70)
        n_val = max(2, int(self.N * 0.15))
        self.train = series.iloc[:n_train].to_numpy()
        self.val = series.iloc[n_train:n_train + n_val].to_numpy()
        self.test = series.iloc[n_train + n_val:].to_numpy()
        self.train_val = series.iloc[:n_train + n_val].to_numpy()

    def run(self):
        print("\n" + "=" * 80)
        print("ЕТАП 1. ДОСЛІДЖЕННЯ ВЛАСТИВОСТЕЙ ЧАСОВОГО РЯДУ")
        print("=" * 80)
        analyzer = TimeSeriesAnalyzer(self.series, self.output_dir)
        props = analyzer.analyze_properties(seasonal_period=self.seasonal_period)
        for k, v in props.items():
            print(f"  {k:22s}: {v}")

        print("\n" + "=" * 80)
        print("ЕТАП 2. R&D ОПТИМІЗАЦІЯ ПАРАМЕТРІВ АЛГОРИТМІВ")
        print("=" * 80)
        window_grid = [3, 5, 7, 10, 14, 21, 30]
        opt_sma_w, sma_rmse, sma_hist = MovingAverageModels.optimize_window_rd(
            self.train, self.val, "SMA", window_grid, trend_damping=False
        )
        opt_wma_w, wma_rmse, wma_hist = MovingAverageModels.optimize_window_rd(
            self.train, self.val, "WMA", window_grid, trend_damping=False
        )
        opt_ema_w, ema_rmse, ema_hist = MovingAverageModels.optimize_window_rd(
            self.train, self.val, "EMA", window_grid, trend_damping=False
        )
        opt_ema_trend_w, ema_tr_rmse, _ = MovingAverageModels.optimize_window_rd(
            self.train, self.val, "EMA", window_grid, trend_damping=True
        )
        print(f"  Оптимальне вікно SMA      : w={opt_sma_w} (Val RMSE={sma_rmse:.4f})")
        print(f"  Оптимальне вікно WMA      : w={opt_wma_w} (Val RMSE={wma_rmse:.4f})")
        print(f"  Оптимальне вікно EMA      : w={opt_ema_w} (Val RMSE={ema_rmse:.4f})")
        print(f"  Оптимальне вікно EMA+Trend: w={opt_ema_trend_w} (Val RMSE={ema_tr_rmse:.4f})")

        best_arima_order, arima_grid = AutoRegressiveModels.grid_search_arima_rd(
            self.train, self.val, p_range=range(0, 4), d_range=range(0, 2), q_range=range(0, 4)
        )
        print(f"  Оптимальна ARIMA(p,d,q)   : {best_arima_order}")
        if not arima_grid.empty:
            print("\n  Топ-5 конфігурацій ARIMA за R&D Grid Search:")
            print(arima_grid.head(5).to_string(index=False))

        self._plot_rd_optimization(sma_hist, wma_hist, ema_hist)

        print("\n" + "=" * 80)
        print("ЕТАП 3. ОЦІНКА МОДЕЛЕЙ НА ТЕСТОВІЙ ВИБІРЦІ ТА ПОРІВНЯННЯ KPI")
        print("=" * 80)
        metrics_list: List[ModelMetrics] = []
        test_preds: Dict[str, np.ndarray] = {}
        n_test = len(self.test)

        t0 = time.perf_counter()
        p_sma_base = MovingAverageModels.forecast_recursive(self.train_val, n_test, "SMA", window=7)
        metrics_list.append(evaluate_predictions(
            self.test, p_sma_base, "SMA (Baseline)", "Ковзне вікно", "w=7", time.perf_counter() - t0
        ))
        test_preds["SMA (Baseline w=7)"] = p_sma_base

        t0 = time.perf_counter()
        p_sma_opt = MovingAverageModels.forecast_recursive(self.train_val, n_test, "SMA", window=opt_sma_w)
        metrics_list.append(evaluate_predictions(
            self.test, p_sma_opt, "SMA (R&D Opt)", "Ковзне вікно", f"w={opt_sma_w}", time.perf_counter() - t0
        ))

        t0 = time.perf_counter()
        p_wma_opt = MovingAverageModels.forecast_recursive(self.train_val, n_test, "WMA", window=opt_wma_w)
        metrics_list.append(evaluate_predictions(
            self.test, p_wma_opt, "WMA (R&D Opt)", "Ковзне вікно", f"w={opt_wma_w}", time.perf_counter() - t0
        ))
        test_preds[f"WMA (w={opt_wma_w})"] = p_wma_opt

        t0 = time.perf_counter()
        p_ema_opt = MovingAverageModels.forecast_recursive(self.train_val, n_test, "EMA", window=opt_ema_w)
        metrics_list.append(evaluate_predictions(
            self.test, p_ema_opt, "EMA (R&D Opt)", "Ковзне вікно", f"w={opt_ema_w}", time.perf_counter() - t0
        ))

        t0 = time.perf_counter()
        p_ema_tr = MovingAverageModels.forecast_recursive(
            self.train_val, n_test, "EMA", window=opt_ema_trend_w, trend_damping=True
        )
        metrics_list.append(evaluate_predictions(
            self.test, p_ema_tr, "EMA + Damped Trend", "Ковзне вікно (R&D)",
            f"w={opt_ema_trend_w}", time.perf_counter() - t0
        ))
        test_preds[f"EMA+Trend (w={opt_ema_trend_w})"] = p_ema_tr

        t0 = time.perf_counter()
        p_arma, aic_arma, bic_arma, _ = AutoRegressiveModels.fit_and_predict_arima(
            self.train_val, n_test, order=(2, 0, 2)
        )
        metrics_list.append(evaluate_predictions(
            self.test, p_arma, "ARMA(2,2)", "Автокореляційна", "(2,0,2)",
            time.perf_counter() - t0, aic=aic_arma, bic=bic_arma
        ))
        test_preds["ARMA(2,2)"] = p_arma

        t0 = time.perf_counter()
        p_arima, aic_arima, bic_arima, fitted_arima = AutoRegressiveModels.fit_and_predict_arima(
            self.train_val, n_test, order=best_arima_order
        )
        metrics_list.append(evaluate_predictions(
            self.test, p_arima, "ARIMA (R&D Opt)", "Автокореляційна", str(best_arima_order),
            time.perf_counter() - t0, aic=aic_arima, bic=bic_arima
        ))
        test_preds[f"ARIMA{best_arima_order}"] = p_arima

        t0 = time.perf_counter()
        p_hw, aic_hw, bic_hw = StatisticalLearningModels.fit_predict_holtwinters(
            self.train_val, n_test, seasonal_period=self.seasonal_period
        )
        metrics_list.append(evaluate_predictions(
            self.test, p_hw, "Holt-Winters", "Стат. навчання", f"add/add, s={self.seasonal_period}",
            time.perf_counter() - t0, aic=aic_hw, bic=bic_hw
        ))
        test_preds["Holt-Winters"] = p_hw

        t0 = time.perf_counter()
        p_ridge = StatisticalLearningModels.fit_predict_ridge_lags(
            self.train_val, n_test, n_lags=14, alpha=10.0
        )
        metrics_list.append(evaluate_predictions(
            self.test, p_ridge, "Ridge (Lags+Trend)", "Стат. навчання (ML)", "lags=14, alpha=10",
            time.perf_counter() - t0
        ))
        test_preds["Ridge (Lags+Trend)"] = p_ridge

        df_kpi = pd.DataFrame([m.__dict__ for m in metrics_list])
        print(df_kpi.to_string(index=False))

        self._plot_test_comparison(test_preds)
        self._plot_residuals(fitted_arima)

        print("\n" + "=" * 80)
        print("ЕТАП 4. ЕКСТРАПОЛЯЦІЯ НА 0.5, 1.0, 1.5 ТА 2.0 ІНТЕРВАЛУ СПОСТЕРЕЖЕННЯ")
        print("=" * 80)
        self.run_extrapolation(opt_ema_trend_w, best_arima_order)

    def run_extrapolation(self, opt_ema_w: int, best_arima_order: Tuple[int, int, int]):
        horizons = [0.5, 1.0, 1.5, 2.0]
        full_data = self.series.to_numpy()
        max_steps = int(round(2.0 * self.N))

        ext_sma = MovingAverageModels.forecast_recursive(full_data, max_steps, "SMA", window=opt_ema_w)
        ext_ema_tr = MovingAverageModels.forecast_recursive(
            full_data, max_steps, "EMA", window=opt_ema_w, trend_damping=True
        )
        ext_arima, _, _, fitted_full = AutoRegressiveModels.fit_and_predict_arima(
            full_data, max_steps, order=best_arima_order
        )
        ext_hw, _, _ = StatisticalLearningModels.fit_predict_holtwinters(
            full_data, max_steps, seasonal_period=self.seasonal_period
        )
        ext_ridge = StatisticalLearningModels.fit_predict_ridge_lags(
            full_data, max_steps, n_lags=14, alpha=10.0
        )

        forecast_obj = fitted_full.get_forecast(steps=max_steps)
        conf_int = np.asarray(forecast_obj.conf_int(alpha=0.05))

        summary_rows = []
        for h in horizons:
            k = int(round(h * self.N))
            summary_rows.append({
                "Горизонт": f"{h:.1f}*N ({k} кроків)",
                "SMA_кінцеве": round(float(ext_sma[k - 1]), 4),
                "EMA+Trend_кінцеве": round(float(ext_ema_tr[k - 1]), 4),
                "ARIMA_кінцеве": round(float(ext_arima[k - 1]), 4),
                "HoltWinters_кінцеве": round(float(ext_hw[k - 1]), 4),
                "Ridge_кінцеве": round(float(ext_ridge[k - 1]), 4),
                "ARIMA_95%_CI_ширина": round(float(conf_int[k - 1, 1] - conf_int[k - 1, 0]), 4),
            })

        df_ext = pd.DataFrame(summary_rows)
        print(df_ext.to_string(index=False))

        freq = pd.infer_freq(self.series.index) or "D"
        future_index = pd.date_range(start=self.series.index[-1], periods=max_steps + 1, freq=freq)[1:]

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(self.series.index, full_data, label="Спостережувані дані (N)", color="black", linewidth=2)
        ax.plot(future_index, ext_sma, label=f"SMA (w={opt_ema_w})", linestyle=":", alpha=0.8)
        ax.plot(future_index, ext_ema_tr, label=f"EMA + Damped Trend (w={opt_ema_w})", linestyle="--")
        ax.plot(future_index, ext_arima, label=f"ARIMA{best_arima_order}", linewidth=2)
        ax.plot(future_index, ext_hw, label="Holt-Winters", linestyle="-.")
        ax.plot(future_index, ext_ridge, label="Ridge (Lags+Trend)", alpha=0.85)
        ax.fill_between(
            future_index, conf_int[:, 0], conf_int[:, 1],
            color="tab:blue", alpha=0.12, label="95% довірчий інтервал ARIMA"
        )

        for h in horizons:
            idx_pos = int(round(h * self.N)) - 1
            x_marker = future_index[idx_pos]
            ax.axvline(x_marker, color="red", linestyle="--", alpha=0.4)
            ax.text(x_marker, ax.get_ylim()[1] * 0.99, f" {h}N", color="red", fontsize=9, va="top")

        ax.set_title("Екстраполяція часового ряду на горизонти 0.5N, 1.0N, 1.5N, 2.0N")
        ax.set_xlabel("Час")
        ax.set_ylabel(self.series.name or "Значення")
        ax.legend(loc="upper left", ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "05_extrapolation_horizons.png"), dpi=150)
        plt.close(fig)

    def _plot_rd_optimization(self, sma_hist, wma_hist, ema_hist):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot([x[0] for x in sma_hist], [x[1] for x in sma_hist], marker="o", label="SMA")
        ax.plot([x[0] for x in wma_hist], [x[1] for x in wma_hist], marker="s", label="WMA")
        ax.plot([x[0] for x in ema_hist], [x[1] for x in ema_hist], marker="^", label="EMA")
        ax.set_title("R&D оптимізація ширини ковзного вікна w за метрикою Val RMSE")
        ax.set_xlabel("Розмір вікна w")
        ax.set_ylabel("Validation RMSE")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "03_rd_window_optimization.png"), dpi=150)
        plt.close(fig)

    def _plot_test_comparison(self, test_preds: Dict[str, np.ndarray]):
        test_index = self.series.index[-len(self.test):]
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(test_index, self.test, label="Фактичні значення (Test)", color="black", linewidth=2.2)
        for name, preds in test_preds.items():
            ax.plot(test_index, preds, label=name, alpha=0.85)
        ax.set_title("Порівняння моделей на тестовому інтервалі")
        ax.set_xlabel("Дата")
        ax.set_ylabel(self.series.name or "Значення")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "04_test_comparison.png"), dpi=150)
        plt.close(fig)

    def _plot_residuals(self, fitted_arima):
        residuals = fitted_arima.resid
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(residuals)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_title("Залишки оптимальної ARIMA у часі")
        axes[1].hist(residuals, bins=25, edgecolor="black", alpha=0.75)
        axes[1].set_title("Розподіл залишків ARIMA")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "06_arima_residuals.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "lab2" / "data2.csv"
    PLOTS_DIR = BASE_DIR / "plots"

    series = TimeSeriesLoader.load_data(
        file_path=str(DATA_PATH),
        date_col="Timestamp",
        value_col="Value",
    )

    pipeline = TimeSeriesPipeline(series=series, output_dir=str(PLOTS_DIR))
    pipeline.run()
    print(f"\n[DONE] Усі графіки збережено у папку '{PLOTS_DIR}'.")