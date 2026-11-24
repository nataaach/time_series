from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


@dataclass
class TimeSeriesDataset:
    name: str
    domain: str
    dates: pd.DatetimeIndex
    values: np.ndarray
    seasonal_period: Optional[int]
    description: str


class TimeSeriesLoader:
    """Клас для завантаження часових рядів із CSV та генерації додаткових R&D-наборів."""

    @staticmethod
    def _detect_seasonal_period(dates: pd.DatetimeIndex, values: np.ndarray) -> Optional[int]:
        n = len(values)
        if n < 20:
            return None

        diff_vals = np.diff(values)
        t = np.arange(len(diff_vals), dtype=float)
        detrended = diff_vals - np.polyval(np.polyfit(t, diff_vals, 1), t)
        if np.std(detrended) < 1e-12:
            return None

        max_lag = min(60, len(detrended) // 3)
        if max_lag < 5:
            return None

        acf_vals = np.nan_to_num(acf(detrended, nlags=max_lag, fft=True), nan=0.0)

        def is_local_peak(lag: int) -> bool:
            left = acf_vals[lag - 1] if lag - 1 >= 1 else 0.0
            right = acf_vals[lag + 1] if lag + 1 <= max_lag else 0.0
            return acf_vals[lag] > 0.35 and acf_vals[lag] > left + 0.05 and acf_vals[lag] > right + 0.05

        freq = pd.infer_freq(dates)
        freq_map = {
            "h": 24, "H": 24, "60min": 24,
            "D": 7, "B": 5,
            "MS": 12, "M": 12, "ME": 12,
            "QS": 4, "Q": 4, "QE": 4,
        }
        if freq:
            for k, p in freq_map.items():
                if freq.startswith(k) and p < max_lag and is_local_peak(p):
                    return p

        for lag in range(4, max_lag):
            if is_local_peak(lag):
                return lag
        return None

    @staticmethod
    def load_data(
        file_path: str,
        date_col: str = "Timestamp",
        value_col: str = "Value",
        name: str = "Lab2_Data2",
        domain: str = "Власні дані (ЛР2)",
        seasonal_period: Optional[int] = None,
    ) -> TimeSeriesDataset:
        df = pd.read_csv(file_path)
        if date_col not in df.columns or value_col not in df.columns:
            raise ValueError(
                f"У файлі {file_path} очікуються колонки '{date_col}' та '{value_col}'. "
                f"Наявні колонки: {list(df.columns)}"
            )

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
        df = df.drop_duplicates(subset=[date_col])

        dates = pd.DatetimeIndex(df[date_col])
        values = df[value_col].to_numpy(dtype=float)

        if seasonal_period is None:
            seasonal_period = TimeSeriesLoader._detect_seasonal_period(dates, values)

        return TimeSeriesDataset(
            name=name,
            domain=domain,
            dates=dates,
            values=values,
            seasonal_period=seasonal_period,
            description=f"Дані з {file_path} ({len(values)} точок, період сезонності={seasonal_period})",
        )

    @staticmethod
    def generate_domain_datasets(seed: int = 42) -> list[TimeSeriesDataset]:
        """Додаткові еталонні ряди для R&D-верифікації селектора (Група вимог 3)."""
        rng = np.random.default_rng(seed)

        n1 = 240
        dates1 = pd.date_range("2025-01-01", periods=n1, freq="D")
        shocks = rng.normal(0.0, 0.14, size=n1)
        drift = np.concatenate([np.full(80, 0.015), np.full(80, -0.010), np.full(80, 0.032)])
        y1 = 38.5 + np.cumsum(drift + shocks)
        ds1 = TimeSeriesDataset(
            name="FX_USD_UAH",
            domain="Валютний трейдинг",
            dates=dates1,
            values=y1,
            seasonal_period=None,
            description="Динаміка курсу USD/UAH (структурні зсуви тренду)",
        )

        n2 = 120
        dates2 = pd.date_range("2016-01-01", periods=n2, freq="MS")
        t2 = np.arange(n2, dtype=float)
        y2 = (
            24.0
            + 0.025 * t2
            + 11.5 * np.sin(2 * np.pi * t2 / 12.0 - np.pi / 2)
            + 1.8 * np.cos(4 * np.pi * t2 / 12.0)
            + rng.normal(0.0, 0.85, size=n2)
        )
        ds2 = TimeSeriesDataset(
            name="Meteo_Temperature",
            domain="Метеорологія",
            dates=dates2,
            values=y2,
            seasonal_period=12,
            description="Сезонний метеорологічний ряд (період 12)",
        )

        n3 = 180
        dates3 = pd.date_range("2025-06-01", periods=n3, freq="D")
        t3 = np.arange(n3, dtype=float)
        y3 = (
            100.0
            + 0.35 * t3
            + 0.014 * (t3 ** 2)
            + 12.0 * np.sin(2 * np.pi * t3 / 7.0)
            + rng.normal(0.0, 4.5, size=n3)
        )
        outlier_idx = rng.choice(np.arange(20, n3 - 20), size=6, replace=False)
        y3[outlier_idx] += rng.choice([-1, 1], size=6) * rng.uniform(30.0, 48.0, size=6)
        ds3 = TimeSeriesDataset(
            name="BigData_Workload",
            domain="Моніторинг Big Data платформи",
            dates=dates3,
            values=y3,
            seasonal_period=7,
            description="Нелінійний квадратичний тренд + викиди",
        )

        return [ds1, ds2, ds3]


DataLoader = TimeSeriesLoader