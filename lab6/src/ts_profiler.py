import warnings
from typing import Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose


class TimeSeriesProfiler:
    """Модуль статистичного аналізу та виявлення властивостей часового ряду."""

    @staticmethod
    def profile(y: np.ndarray, seasonal_period: Optional[int] = None) -> dict:
        y = np.asarray(y, dtype=float)
        n = len(y)
        mean_val = float(np.mean(y))
        std_val = float(np.std(y, ddof=1)) if n > 1 else 0.0
        cv = float(std_val / (abs(mean_val) + 1e-9))

        # 1. Тест Дікі-Фуллера (ADF)
        if np.std(y) < 1e-12 or n < 10:
            adf_stat, adf_pvalue = 0.0, 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_res = adfuller(y, autolag="AIC")
            adf_stat, adf_pvalue = float(adf_res[0]), float(adf_res[1])
        is_stationary = adf_pvalue < 0.05

        # 2. Декомпозиція та сила тренду F_T / сезонності F_S
        trend_strength = 0.0
        seasonality_strength = 0.0

        if (
            seasonal_period is not None
            and seasonal_period >= 2
            and n >= 2 * seasonal_period
            and np.std(y) > 1e-12
        ):
            decomp = seasonal_decompose(
                y, model="additive", period=seasonal_period, extrapolate_trend="period"
            )
            t_comp = np.asarray(decomp.trend)
            s_comp = np.asarray(decomp.seasonal)
            r_comp = np.asarray(decomp.resid)

            var_r = np.var(r_comp)
            var_tr = np.var(t_comp + r_comp)
            var_sr = np.var(s_comp + r_comp)
            trend_strength = float(max(0.0, 1.0 - var_r / (var_tr + 1e-12)))
            seasonality_strength = float(max(0.0, 1.0 - var_r / (var_sr + 1e-12)))
        else:
            win = max(3, min(15, n // 6 if n >= 18 else 3))
            kernel = np.ones(win) / win
            smooth = np.convolve(y, kernel, mode="same")
            resid = y - smooth
            trend_strength = float(max(0.0, 1.0 - np.var(resid) / (np.var(y) + 1e-12)))

        # 3. Індекс нелінійності тренду та якість поліноміальної апроксимації (R^2_poly2)
        t = np.arange(n, dtype=float)
        t_norm = (t - t.mean()) / (t.std() + 1e-9)
        p1 = np.polyval(np.polyfit(t_norm, y, 1), t_norm)
        p2 = np.polyval(np.polyfit(t_norm, y, 2), t_norm)
        ss_tot = np.sum((y - mean_val) ** 2) + 1e-12
        r2_lin = float(1.0 - np.sum((y - p1) ** 2) / ss_tot)
        r2_poly2 = float(1.0 - np.sum((y - p2) ** 2) / ss_tot)
        nonlinearity_index = float(max(0.0, r2_poly2 - r2_lin))

        # 4. Індекс структурних зламів локального нахилу тренду
        seg_size = max(5, n // 3)
        slopes = []
        for i in range(3):
            start = i * seg_size
            end = (i + 1) * seg_size if i < 2 else n
            if end - start >= 3:
                ti = np.arange(end - start, dtype=float)
                slopes.append(float(np.polyfit(ti, y[start:end], 1)[0]))
        if len(slopes) >= 2:
            slope_disp = np.std(slopes) / (abs(np.mean(slopes)) + std_val / max(n, 1) + 1e-9)
            structural_shift_index = float(slope_disp / (1.0 + slope_disp))
        else:
            structural_shift_index = 0.0

        # 5. Частка точкових викидів (за локальною ковзною медіаною w=5 та MAD)
        local_med = pd.Series(y).rolling(window=5, center=True, min_periods=1).median().to_numpy()
        spike_resid = y - local_med
        med = np.median(spike_resid)
        mad = np.median(np.abs(spike_resid - med))
        robust_sigma = 1.4826 * mad + 1e-9
        outlier_mask = np.abs(spike_resid - med) > 4.0 * robust_sigma
        outlier_ratio = float(np.mean(outlier_mask))

        # 6. Автокореляція
        if np.std(y) < 1e-12 or n < 5:
            acf_lag1, acf_lag2 = 0.0, 0.0
        else:
            nlags = min(5, max(2, n // 4))
            acf_vals = np.nan_to_num(acf(y, nlags=nlags, fft=False), nan=0.0)
            acf_lag1 = float(acf_vals[1]) if len(acf_vals) > 1 else 0.0
            acf_lag2 = float(acf_vals[2]) if len(acf_vals) > 2 else 0.0

        return {
            "n": n,
            "mean": mean_val,
            "std": std_val,
            "cv": cv,
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pvalue,
            "is_stationary": is_stationary,
            "trend_strength": trend_strength,
            "seasonality_strength": seasonality_strength,
            "r2_poly2": r2_poly2,
            "nonlinearity_index": nonlinearity_index,
            "structural_shift_index": structural_shift_index,
            "outlier_ratio": outlier_ratio,
            "acf_lag1": acf_lag1,
            "acf_lag2": acf_lag2,
        }