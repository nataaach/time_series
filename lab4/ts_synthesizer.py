"""
Модуль генерації синтетичних часових рядів та доведення адекватності.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class TSSynthesizer:
    def __init__(self, length: int, slope: float, intercept: float,
                 period: int, amplitude: float, noise_std: float):
        self.length = length
        self.slope = slope
        self.intercept = intercept
        self.period = period
        self.amplitude = amplitude
        self.noise_std = noise_std

    def generate(self, seed: int = 42) -> pd.Series:
        """Синтез модельного ряду з виявленими закономірностями."""
        np.random.seed(seed)
        t = np.arange(self.length)
        trend = self.intercept + self.slope * t
        seasonality = self.amplitude * np.sin(2 * np.pi * t / self.period)
        noise = np.random.normal(0, self.noise_std, size=self.length)
        
        # Виручка не може бути від'ємною
        synthetic_values = np.clip(trend + seasonality + noise, 0, None)
        return pd.Series(synthetic_values)

    @staticmethod
    def verify_adequacy(real_series: pd.Series, synthetic_series: pd.Series) -> dict:
        """Верифікація адекватності моделі критерієм Колмогорова-Смірнова."""
        r = real_series.values
        s = synthetic_series.values
        ks_stat, ks_pval = ks_2samp(r, s)
        
        mean_err = abs(np.mean(r) - np.mean(s)) / (np.mean(r) + 1e-6)
        std_err = abs(np.std(r) - np.std(s)) / (np.std(r) + 1e-6)

        return {
            'ks_stat': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'is_adequate': bool(ks_pval >= 0.05),
            'mean_rel_error': float(mean_err),
            'std_rel_error': float(std_err)
        }