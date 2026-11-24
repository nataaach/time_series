import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


class TimeSeriesVisualizer:
    """Побудова та збереження графіків для розділу 5.3.4 звіту."""

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_eda_and_decomposition(self, ds, train_len: int) -> str:
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        axes[0].plot(ds.dates[:train_len], ds.values[:train_len], label="Train (навчальна вибірка)", color="#1f77b4")
        axes[0].plot(ds.dates[train_len:], ds.values[train_len:], label="Test (контрольний горизонт)", color="#ff7f0e")
        axes[0].axvline(ds.dates[train_len - 1], color="gray", linestyle="--", linewidth=1.2)
        axes[0].set_title(f"[{ds.name}] Вхідний часовий ряд ({ds.domain})")
        axes[0].set_ylabel("Значення")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        if ds.seasonal_period and len(ds.values) >= 2 * ds.seasonal_period:
            dec = seasonal_decompose(ds.values, model="additive", period=ds.seasonal_period, extrapolate_trend="period")
            axes[1].plot(ds.dates, dec.trend, label="Тренд (T)", color="#2ca02c", linewidth=2)
            axes[1].plot(ds.dates, dec.seasonal + np.mean(dec.trend), label="Сезонність + рівень", color="#9467bd", alpha=0.75)
            axes[1].set_title(f"Адитивна декомпозиція (період = {ds.seasonal_period})")
        else:
            s = pd.Series(ds.values)
            roll_mean = s.rolling(window=10, min_periods=1).mean()
            roll_std = s.rolling(window=10, min_periods=1).std().fillna(0)
            axes[1].plot(ds.dates, roll_mean, label="Ковзне середнє (w=10)", color="#2ca02c", linewidth=2)
            axes[1].fill_between(ds.dates, roll_mean - roll_std, roll_mean + roll_std, alpha=0.2, color="#2ca02c", label="±1 STD")
            axes[1].set_title("Ковзні статистики (тренд та локальна волатильність)")
        axes[1].set_ylabel("Компоненти")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{ds.name}_1_eda.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_comparison_forecast(
        self,
        ds,
        train_len: int,
        es_res: dict,
        reg_res: dict,
        future_dates: pd.DatetimeIndex,
    ) -> str:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.2, 1]})

        train_dates = ds.dates[:train_len]
        test_dates = ds.dates[train_len:]
        train_y = ds.values[:train_len]
        test_y = ds.values[train_len:]

        ax = axes[0]
        ax.plot(train_dates, train_y, label="Факт (Train)", color="black", alpha=0.65, linewidth=1.3)
        ax.plot(test_dates, test_y, label="Факт (Test)", color="black", linewidth=2.2, marker="o", markersize=3)

        ax.plot(train_dates, es_res["fitted_train"], "--", color="#1f77b4", alpha=0.75, label=f"ES fit ({es_res['best_model_name']})")
        ax.plot(test_dates, es_res["test_pred"], color="#1f77b4", linewidth=2.2, label="ES тест-прогноз")
        ax.plot(future_dates, es_res["future_pred"], ":", color="#1f77b4", linewidth=2.2, label="ES екстраполяція")

        ax.plot(train_dates, reg_res["fitted_train"], "--", color="#d62728", alpha=0.75, label=f"Reg fit ({reg_res['best_model_name']})")
        ax.plot(test_dates, reg_res["test_pred"], color="#d62728", linewidth=2.2, label="Reg тест-прогноз")
        ax.plot(future_dates, reg_res["future_pred"], ":", color="#d62728", linewidth=2.2, label="Reg екстраполяція")

        ax.axvline(train_dates[-1], color="gray", linestyle="--", linewidth=1.1)
        ax.axvline(test_dates[-1], color="purple", linestyle="-.", linewidth=1.1)
        ax.set_title(f"[{ds.name}] Порівняння згладжування та екстраполяції: Exponential Smoothing vs Regression")
        ax.set_ylabel("Значення")
        ax.legend(loc="best", ncol=2, fontsize=8.5)
        ax.grid(alpha=0.3)

        ax_err = axes[1]
        err_es = test_y - es_res["test_pred"]
        err_reg = test_y - reg_res["test_pred"]
        ax_err.plot(test_dates, err_es, marker="o", markersize=3, color="#1f77b4", label="Залишки ES (Test)")
        ax_err.plot(test_dates, err_reg, marker="s", markersize=3, color="#d62728", label="Залишки Reg (Test)")
        ax_err.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax_err.set_title("Похибки екстраполяції (y_true - y_pred) на тестовій вибірці")
        ax_err.set_ylabel("Похибка")
        ax_err.legend()
        ax_err.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{ds.name}_2_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_kpi_benchmark(self, summary_df: pd.DataFrame) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        datasets = summary_df["Dataset"].unique()
        x = np.arange(len(datasets))
        w = 0.35

        es_rmse = [
            summary_df[(summary_df["Dataset"] == d) & (summary_df["Paradigm"] == "ExponentialSmoothing")]["Test_RMSE"].values[0]
            for d in datasets
        ]
        reg_rmse = [
            summary_df[(summary_df["Dataset"] == d) & (summary_df["Paradigm"] == "Regression")]["Test_RMSE"].values[0]
            for d in datasets
        ]

        axes[0].bar(x - w / 2, es_rmse, width=w, label="Exponential Smoothing", color="#1f77b4")
        axes[0].bar(x + w / 2, reg_rmse, width=w, label="Regression", color="#d62728")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(datasets, rotation=12)
        axes[0].set_title("Порівняння RMSE на екстраполяції (Test)")
        axes[0].set_ylabel("RMSE (менше = краще)")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        es_mape = [
            summary_df[(summary_df["Dataset"] == d) & (summary_df["Paradigm"] == "ExponentialSmoothing")]["Test_MAPE_%"].values[0]
            for d in datasets
        ]
        reg_mape = [
            summary_df[(summary_df["Dataset"] == d) & (summary_df["Paradigm"] == "Regression")]["Test_MAPE_%"].values[0]
            for d in datasets
        ]

        axes[1].bar(x - w / 2, es_mape, width=w, label="Exponential Smoothing", color="#1f77b4")
        axes[1].bar(x + w / 2, reg_mape, width=w, label="Regression", color="#d62728")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(datasets, rotation=12)
        axes[1].set_title("Порівняння MAPE (%) на екстраполяції (Test)")
        axes[1].set_ylabel("MAPE, %")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "benchmark_kpi_summary.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path