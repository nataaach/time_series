import os
import random
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import tensorflow as tf
from keras import layers, models, callbacks, optimizers


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


@dataclass
class Config:
    output_dir: str = "output"
    window_size: int = 10
    horizon: int = 5
    seasonal_period: int = 7
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    epochs: int = 70
    batch_size: int = 8
    learning_rate: float = 1e-3
    mc_samples: int = 30


class MCDropout(layers.Dropout):
    """Шар Dropout з керованим режимом Monte Carlo на інференсі без впливу на BatchNorm."""

    def __init__(self, rate: float, **kwargs):
        super().__init__(rate, **kwargs)
        self.mc_active = tf.Variable(False, trainable=False, dtype=tf.bool)

    def call(self, inputs, training=None):
        return tf.cond(
            self.mc_active,
            lambda: super(MCDropout, self).call(inputs, training=True),
            lambda: super(MCDropout, self).call(inputs, training=training),
        )


class TimeSeriesDataLoader:
    @staticmethod
    def load_data(csv_path: str = None, value_col: str = "Value") -> pd.DataFrame:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not csv_path:
            csv_path = os.path.join(base_dir, "data7.csv")
        elif not os.path.isabs(csv_path) and not os.path.exists(csv_path):
            csv_path = os.path.join(base_dir, csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Файл даних не знайдено: {csv_path}")

        print(f"[DATA] Завантаження локального CSV: {csv_path}")
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
        df.columns = [str(c).strip() for c in df.columns]

        date_candidates = ["timestamp", "date", "datetime", "time", "дата"]
        date_col = next((c for c in df.columns if c.lower() in date_candidates), None)

        if date_col is not None:
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=["date"])
        else:
            df["date"] = pd.date_range(start="2026-06-14", periods=len(df), freq="D")

        if value_col not in df.columns:
            matched_col = next(
                (c for c in df.columns if c.lower() == value_col.lower()), None
            )
            if matched_col is None:
                raise ValueError(
                    f"Колонку '{value_col}' не знайдено. Доступні колонки: {list(df.columns)}"
                )
            value_col = matched_col

        df["value"] = (
            df[value_col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = (
            df[["date", "value"]]
            .dropna()
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )
        print(
            f"[DATA] Успішно завантажено {len(df)} спостережень "
            f"({df['date'].min().date()} — {df['date'].max().date()})."
        )
        return df


class TimeSeriesAnalyzer:
    """Клас для дослідження властивостей, декомпозиції та препроцесінгу даних."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = StandardScaler()

    def preprocess_and_eda(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        df = df.set_index("date").asfreq("D")
        df["value"] = df["value"].interpolate(method="time").bfill().ffill()
        df = df.reset_index()

        series = df["value"].values
        adf_orig = adfuller(series)
        adf_diff = adfuller(np.diff(series))

        stl = STL(df["value"], period=self.cfg.seasonal_period, robust=True)
        res = stl.fit()
        df["trend"] = res.trend
        df["seasonal"] = res.seasonal
        df["resid"] = res.resid

        stats_info = {
            "N": len(df),
            "mean": float(np.mean(series)),
            "std": float(np.std(series)),
            "min": float(np.min(series)),
            "max": float(np.max(series)),
            "adf_stat_orig": float(adf_orig[0]),
            "adf_p_orig": float(adf_orig[1]),
            "adf_stat_diff": float(adf_diff[0]),
            "adf_p_diff": float(adf_diff[1]),
        }
        self._plot_eda(df, stats_info)
        return df, stats_info

    def _plot_eda(self, df: pd.DataFrame, stats: dict):
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(df["date"], df["value"], color="#1f77b4", lw=1.5)
        axes[0].set_title(
            f"Вихідний часовий ряд (N={stats['N']}, ADF p={stats['adf_p_orig']:.4f} -> "
            f"1-ша різниця ADF p={stats['adf_p_diff']:.4e})"
        )
        axes[0].set_ylabel("Значення")

        axes[1].plot(df["date"], df["trend"], color="#d62728", lw=1.5)
        axes[1].set_title("Трендова компонента (STL)")
        axes[1].set_ylabel("Тренд")

        axes[2].plot(df["date"], df["seasonal"], color="#2ca02c", lw=1.2)
        axes[2].set_title(f"Сезонна компонента (період={self.cfg.seasonal_period})")
        axes[2].set_ylabel("Сезонність")

        axes[3].plot(df["date"], df["resid"], color="#7f7f7f", lw=1.0)
        axes[3].axhline(0, color="black", ls="--", lw=0.8)
        axes[3].set_title("Залишкова (стохастична) компонента")
        axes[3].set_ylabel("Залишки")

        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.cfg.output_dir, "01_eda_decomposition.png"), dpi=200
        )
        plt.close()

        max_lags = min(30, len(df) // 2 - 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(df["value"], lags=max_lags, ax=axes[0], title="ACF вихідного ряду")
        plot_pacf(df["value"], lags=max_lags, ax=axes[1], title="PACF вихідного ряду")
        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, "02_acf_pacf.png"), dpi=200)
        plt.close()

    def build_features_and_windows(
        self, df: pd.DataFrame
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
    ]:
        """Формування масштабованих ознак [value_scaled, sin_season, cos_season] та вікон."""
        n = len(df)
        train_end = int(round(n * self.cfg.train_ratio))
        val_end = int(round(n * (self.cfg.train_ratio + self.cfg.val_ratio)))

        values = df["value"].values.reshape(-1, 1)
        self.scaler.fit(values[:train_end])
        scaled_vals = self.scaler.transform(values).flatten()

        t_idx = np.arange(n)
        sin_p = np.sin(2 * np.pi * t_idx / self.cfg.seasonal_period)
        cos_p = np.cos(2 * np.pi * t_idx / self.cfg.seasonal_period)
        features = np.column_stack([scaled_vals, sin_p, cos_p])

        W, H = self.cfg.window_size, self.cfg.horizon
        X_all, y_all, target_end_idx = [], [], []
        for i in range(W, n - H + 1):
            X_all.append(features[i - W : i])
            y_all.append(scaled_vals[i : i + H])
            target_end_idx.append(i + H - 1)

        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.float32)
        target_end_idx = np.array(target_end_idx)

        train_mask = target_end_idx < train_end
        val_mask = (target_end_idx >= train_end) & (target_end_idx < val_end)
        test_mask = target_end_idx >= val_end

        return (
            X_all[train_mask],
            y_all[train_mask],
            X_all[val_mask],
            y_all[val_mask],
            X_all[test_mask],
            y_all[test_mask],
            features,
            val_end,
        )


class CompositeLossBuilder:
    @staticmethod
    def build(
        alpha: float = 0.40,
        beta: float = 0.25,
        gamma: float = 0.25,
        lam: float = 0.10,
        delta: float = 1.0,
    ):
        def composite_loss(y_true, y_pred):
            error = y_true - y_pred
            abs_error = tf.abs(error)

            huber = tf.where(
                abs_error <= delta,
                0.5 * tf.square(error),
                delta * (abs_error - 0.5 * delta),
            )
            loss_huber = tf.reduce_mean(huber)
            loss_mae = tf.reduce_mean(abs_error)

            if y_true.shape[-1] is not None and y_true.shape[-1] > 1:
                diff_true = y_true[:, 1:] - y_true[:, :-1]
                diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
                loss_diff = tf.reduce_mean(tf.abs(diff_true - diff_pred))

                var_true = tf.math.reduce_variance(y_true, axis=-1)
                var_pred = tf.math.reduce_variance(y_pred, axis=-1)
                std_true = tf.sqrt(var_true + 1e-8)
                std_pred = tf.sqrt(var_pred + 1e-8)
                loss_std = tf.reduce_mean(tf.abs(std_true - std_pred))
            else:
                loss_diff = 0.0
                loss_std = 0.0

            return (
                alpha * loss_huber
                + beta * loss_mae
                + gamma * loss_diff
                + lam * loss_std
            )

        composite_loss.__name__ = "composite_loss"
        return composite_loss


class DeepLearningForecaster:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.input_shape = (cfg.window_size, 3)

    def build_model(self, model_type: str, loss_fn) -> models.Model:
        inp = layers.Input(shape=self.input_shape, name="Input_Window")

        if model_type == "LSTM":
            x = layers.LSTM(64, return_sequences=True, name="LSTM_1_64")(inp)
            x = MCDropout(0.2, name="Dropout_1")(x)
            x = layers.LSTM(32, return_sequences=False, name="LSTM_2_32")(x)
            x = MCDropout(0.2, name="Dropout_2")(x)
            x = layers.Dense(32, activation="relu", name="Dense_32")(x)

        elif model_type == "GRU":
            x = layers.GRU(64, return_sequences=True, name="GRU_1_64")(inp)
            x = MCDropout(0.2, name="Dropout_1")(x)
            x = layers.GRU(32, return_sequences=False, name="GRU_2_32")(x)
            x = MCDropout(0.2, name="Dropout_2")(x)
            x = layers.Dense(32, activation="relu", name="Dense_32")(x)

        elif model_type == "CNN_BiLSTM":
            x = layers.Conv1D(
                filters=48,
                kernel_size=3,
                padding="same",
                activation="relu",
                name="Conv1D_48",
            )(inp)
            x = layers.BatchNormalization(name="BatchNorm")(x)
            x = MCDropout(0.15, name="Dropout_1")(x)
            x = layers.Bidirectional(
                layers.LSTM(32, return_sequences=False),
                name="BiLSTM_32",
            )(x)
            x = MCDropout(0.2, name="Dropout_2")(x)
            x = layers.Dense(32, activation="relu", name="Dense_32")(x)

        else:
            raise ValueError(f"Невідомий тип моделі: {model_type}")

        out = layers.Dense(
            self.cfg.horizon, activation="linear", name="Output_Horizon"
        )(x)
        model = models.Model(inputs=inp, outputs=out, name=f"TS_{model_type}")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.cfg.learning_rate),
            loss=loss_fn,
            metrics=["mae", "mse"],
        )
        return model

    def draw_architecture_diagram(self, model_type: str, filename: str):
        specs = {
            "LSTM": [
                ("Input Layer", f"shape: (batch, {self.cfg.window_size}, 3)", "#e3f2fd"),
                ("LSTM (64 units)", "return_sequences=True, tanh/sigmoid", "#bbdefb"),
                ("MCDropout (0.2)", "Monte Carlo active at inference", "#fff9c4"),
                ("LSTM (32 units)", "return_sequences=False", "#bbdefb"),
                ("MCDropout (0.2)", "Monte Carlo active at inference", "#fff9c4"),
                ("Dense (32, ReLU)", "Нелінійна проєкція ознак", "#c8e6c9"),
                (
                    "Output Dense",
                    f"units={self.cfg.horizon}, activation=linear",
                    "#ffccbc",
                ),
            ],
            "GRU": [
                ("Input Layer", f"shape: (batch, {self.cfg.window_size}, 3)", "#e3f2fd"),
                ("GRU (64 units)", "return_sequences=True, reset/update gates", "#d1c4e9"),
                ("MCDropout (0.2)", "Monte Carlo active at inference", "#fff9c4"),
                ("GRU (32 units)", "return_sequences=False", "#d1c4e9"),
                ("MCDropout (0.2)", "Monte Carlo active at inference", "#fff9c4"),
                ("Dense (32, ReLU)", "Нелінійна проєкція ознак", "#c8e6c9"),
                (
                    "Output Dense",
                    f"units={self.cfg.horizon}, activation=linear",
                    "#ffccbc",
                ),
            ],
            "CNN_BiLSTM": [
                ("Input Layer", f"shape: (batch, {self.cfg.window_size}, 3)", "#e3f2fd"),
                ("Conv1D (48, k=3)", "Виділення локальних патернів, ReLU", "#b2dfdb"),
                (
                    "BatchNorm + MCDropout",
                    "BatchNorm (eval) + MC-регуляризація (0.15)",
                    "#fff9c4",
                ),
                (
                    "Bidirectional LSTM",
                    "2 x 32 units, прямий і зворотний контекст",
                    "#bbdefb",
                ),
                ("MCDropout (0.2)", "Monte Carlo active at inference", "#fff9c4"),
                ("Dense (32, ReLU)", "Нелінійна проєкція ознак", "#c8e6c9"),
                (
                    "Output Dense",
                    f"units={self.cfg.horizon}, activation=linear",
                    "#ffccbc",
                ),
            ],
        }

        blocks = specs[model_type]
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(blocks) * 1.5 + 1)
        ax.axis("off")
        ax.set_title(
            f"Архітектура нейронної мережі: {model_type}",
            fontsize=13,
            fontweight="bold",
        )

        y_top = len(blocks) * 1.5
        for idx, (title, desc, color) in enumerate(blocks):
            y = y_top - idx * 1.45
            box = patches.FancyBboxPatch(
                (1.5, y - 0.55),
                7.0,
                0.95,
                boxstyle="round,pad=0.1,rounding_size=0.15",
                facecolor=color,
                edgecolor="#37474f",
                linewidth=1.4,
            )
            ax.add_patch(box)
            ax.text(
                5.0,
                y + 0.05,
                title,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
            ax.text(
                5.0,
                y - 0.28,
                desc,
                ha="center",
                va="center",
                fontsize=9,
                color="#263238",
            )
            if idx < len(blocks) - 1:
                ax.annotate(
                    "",
                    xy=(5.0, y - 1.02),
                    xytext=(5.0, y - 0.55),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#37474f"),
                )

        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, filename), dpi=200)
        plt.close()

    def train(
        self, model: models.Model, X_tr, y_tr, X_val, y_val
    ) -> callbacks.History:
        es = callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, verbose=0
        )
        rlr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0
        )
        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            callbacks=[es, rlr],
            verbose=0,
        )
        return history

    @staticmethod
    def _set_mc_dropout(model: models.Model, active: bool):
        for layer in model.layers:
            if isinstance(layer, MCDropout):
                layer.mc_active.assign(active)

    def extrapolate_long_term(
        self,
        model: models.Model,
        last_window: np.ndarray,
        start_time_idx: int,
        total_steps: int,
        scaler: StandardScaler,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        W, H = self.cfg.window_size, self.cfg.horizon
        n_blocks = int(np.ceil(total_steps / H))
        all_mc_trajectories = []

        self._set_mc_dropout(model, True)
        try:
            for _ in range(self.cfg.mc_samples):
                curr_win = last_window.copy()
                curr_t = start_time_idx
                traj_scaled = []

                for _b in range(n_blocks):
                    
                    pred_block = (
                        model(curr_win[np.newaxis, ...], training=False)
                        .numpy()
                        .flatten()
                    )
                    traj_scaled.extend(pred_block.tolist())

                    future_t = np.arange(curr_t, curr_t + H)
                    sin_f = np.sin(2 * np.pi * future_t / self.cfg.seasonal_period)
                    cos_f = np.cos(2 * np.pi * future_t / self.cfg.seasonal_period)
                    new_block_features = np.column_stack([pred_block, sin_f, cos_f])

                    combined = np.vstack([curr_win, new_block_features])
                    curr_win = combined[-W:]
                    curr_t += H

                traj_scaled_arr = np.array(traj_scaled[:total_steps]).reshape(-1, 1)
                traj_real = scaler.inverse_transform(traj_scaled_arr).flatten()
                all_mc_trajectories.append(traj_real)
        finally:
            self._set_mc_dropout(model, False)

        mc_arr = np.array(all_mc_trajectories)
        mean_pred = np.mean(mc_arr, axis=0)
        lower_ci = np.percentile(mc_arr, 5, axis=0)
        upper_ci = np.percentile(mc_arr, 95, axis=0)
        return mean_pred, lower_ci, upper_ci


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = (
        np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0
    )
    r2 = r2_score(y_true, y_pred)

    if y_true.ndim == 2 and y_true.shape[1] > 1:
        dir_true = np.sign(y_true[:, 1:] - y_true[:, :-1])
        dir_pred = np.sign(y_pred[:, 1:] - y_pred[:, :-1])
        da = np.mean(dir_true == dir_pred) * 100.0
    else:
        dir_true = np.sign(np.diff(y_true.flatten()))
        dir_pred = np.sign(np.diff(y_pred.flatten()))
        da = np.mean(dir_true == dir_pred) * 100.0

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE_%": float(mape),
        "R2": float(r2),
        "Dir_Acc_%": float(da),
    }


def main():
    default_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data7.csv")
    parser = argparse.ArgumentParser(
        description="Lab 7: Deep Learning Time Series (Level 4)"
    )
    parser.add_argument(
        "--csv", type=str, default=default_csv, help="Шлях до CSV файлу даних"
    )
    parser.add_argument(
        "--col", type=str, default="Value", help="Назва колонки зі значеннями"
    )
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    df = TimeSeriesDataLoader.load_data(args.csv, args.col)
    analyzer = TimeSeriesAnalyzer(cfg)
    df, stats_info = analyzer.preprocess_and_eda(df)
    print("[EDA] Статистики ряду:", stats_info)

    X_tr, y_tr, X_val, y_val, X_te, y_te, all_features, test_start_idx = (
        analyzer.build_features_and_windows(df)
    )
    print(f"[SPLIT] Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

    comp_loss = CompositeLossBuilder.build(
        alpha=0.40, beta=0.25, gamma=0.25, lam=0.10, delta=1.0
    )
    forecaster = DeepLearningForecaster(cfg)

    model_types = ["LSTM", "GRU", "CNN_BiLSTM"]
    arch_files = {
        "LSTM": "03_arch_LSTM.png",
        "GRU": "04_arch_GRU.png",
        "CNN_BiLSTM": "05_arch_CNN_BiLSTM.png",
    }

    results_table = []
    histories = {}
    extrapolations = {}
    test_rollouts = {}

    n_obs = len(df)
    extrap_steps = int(np.ceil(0.5 * n_obs))
    print(
        f"[EXTRAPOLATION] Довжина ряду N={n_obs}, горизонт екстраполяції H_ext={extrap_steps} (>= 0.5*N)."
    )

    test_len = n_obs - test_start_idx
    last_val_window = all_features[test_start_idx - cfg.window_size : test_start_idx]
    last_obs_window = all_features[-cfg.window_size :]

    for m_type in model_types:
        print(f"\n[TRAIN] Навчання моделі {m_type} із Composite Loss...")
        forecaster.draw_architecture_diagram(m_type, arch_files[m_type])
        model = forecaster.build_model(m_type, comp_loss)
        hist = forecaster.train(model, X_tr, y_tr, X_val, y_val)
        histories[m_type] = hist.history

        y_pred_scaled = model.predict(X_te, verbose=0)
        y_te_real = analyzer.scaler.inverse_transform(y_te)
        y_pred_real = analyzer.scaler.inverse_transform(y_pred_scaled)

        metrics = compute_metrics(y_te_real, y_pred_real)
        metrics["Model"] = f"{m_type} (Composite)"
        metrics["Epochs_Run"] = len(hist.history["loss"])
        metrics["Best_Val_Loss"] = float(np.min(hist.history["val_loss"]))
        results_table.append(metrics)

        test_mean, _, _ = forecaster.extrapolate_long_term(
            model, last_val_window, test_start_idx, test_len, analyzer.scaler
        )
        test_rollouts[m_type] = test_mean

        ext_mean, ext_low, ext_high = forecaster.extrapolate_long_term(
            model, last_obs_window, n_obs, extrap_steps, analyzer.scaler
        )
        extrapolations[m_type] = (ext_mean, ext_low, ext_high)

    best_row = min(results_table, key=lambda r: r["RMSE"])
    best_type = best_row["Model"].split(" ")[0]
    print(f"\n[ABLATION] Порівняння {best_type} зі стандартною функцією втрат MSE...")

    mse_model = forecaster.build_model(best_type, "mse")
    mse_hist = forecaster.train(mse_model, X_tr, y_tr, X_val, y_val)
    y_pred_mse_real = analyzer.scaler.inverse_transform(
        mse_model.predict(X_te, verbose=0)
    )
    mse_metrics = compute_metrics(
        analyzer.scaler.inverse_transform(y_te), y_pred_mse_real
    )
    mse_metrics["Model"] = f"{best_type} (Baseline MSE)"
    mse_metrics["Epochs_Run"] = len(mse_hist.history["loss"])
    mse_metrics["Best_Val_Loss"] = float(np.min(mse_hist.history["val_loss"]))
    results_table.append(mse_metrics)

    ext_mse_mean, _, _ = forecaster.extrapolate_long_term(
        mse_model, last_obs_window, n_obs, extrap_steps, analyzer.scaler
    )

    res_df = pd.DataFrame(results_table)[
        [
            "Model",
            "MSE",
            "RMSE",
            "MAE",
            "MAPE_%",
            "R2",
            "Dir_Acc_%",
            "Epochs_Run",
            "Best_Val_Loss",
        ]
    ]
    res_df.to_csv(os.path.join(cfg.output_dir, "metrics_summary.csv"), index=False)

    print("\n================ ПІДСУМКОВА ТАБЛИЦЯ МЕТРИК ================")
    print(res_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, m_type in zip(axes, model_types):
        h = histories[m_type]
        ax.plot(h["loss"], label="Train Composite Loss", lw=1.8)
        ax.plot(h["val_loss"], label="Val Composite Loss", lw=1.8)
        ax.set_title(f"Крива навчання: {m_type}")
        ax.set_xlabel("Епоха")
        ax.set_ylabel("Втрати")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "06_training_history.png"), dpi=200)
    plt.close()

    future_dates = pd.date_range(
        start=df["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=extrap_steps,
        freq="D",
    )
    test_dates = df["date"].iloc[test_start_idx:]

    plt.figure(figsize=(14, 6.5))
    plt.plot(
        df["date"],
        df["value"],
        color="black",
        lw=1.6,
        label="Фактичні дані (інтервал спостереження)",
    )
    plt.axvline(
        df["date"].iloc[test_start_idx],
        color="gray",
        ls="--",
        lw=1.2,
        label="Межа Test",
    )
    plt.axvline(
        df["date"].iloc[-1],
        color="red",
        ls="--",
        lw=1.4,
        label="Межа екстраполяції",
    )

    colors = {"LSTM": "#1f77b4", "GRU": "#2ca02c", "CNN_BiLSTM": "#d62728"}
    for m_type in model_types:
        plt.plot(
            test_dates,
            test_rollouts[m_type],
            color=colors[m_type],
            ls=":",
            lw=1.8,
            label=f"{m_type} (Test rollout)",
        )
        ext_m, ext_l, ext_u = extrapolations[m_type]
        plt.plot(
            future_dates,
            ext_m,
            color=colors[m_type],
            lw=2.0,
            label=f"{m_type} (Екстраполяція {extrap_steps} дн.)",
        )
        if m_type == best_type:
            plt.fill_between(
                future_dates,
                ext_l,
                ext_u,
                color=colors[m_type],
                alpha=0.15,
                label=f"{m_type} 90% MC-CI",
            )

    plt.title(
        f"Прогноз на тестовій ділянці та довготривала екстраполяція (H_ext = {extrap_steps} >= 0.5 * {n_obs})"
    )
    plt.xlabel("Дата")
    plt.ylabel("Значення ряду (Value)")
    plt.legend(loc="best", ncol=2, fontsize=8.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "07_test_and_extrapolation.png"), dpi=200)
    plt.close()

    # 08: Порівняння метрик моделей
    comp_only = res_df[res_df["Model"].str.contains("Composite")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics_to_plot = [
        ("RMSE", "RMSE (нижче = краще)"),
        ("MAPE_%", "MAPE % (нижче = краще)"),
        ("Dir_Acc_%", "Directional Accuracy % (вище = краще)"),
    ]
    for ax, (col, title) in zip(axes, metrics_to_plot):
        bars = ax.bar(
            comp_only["Model"],
            comp_only[col],
            color=["#1f77b4", "#2ca02c", "#d62728"],
            alpha=0.85,
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)
        max_val = float(comp_only[col].max())
        ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1.0)
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "08_metrics_comparison.png"), dpi=200)
    plt.close()

    # 09: Абляційний аналіз функції втрат
    tail_n = min(120, len(df))
    plt.figure(figsize=(13, 5))
    plt.plot(
        df["date"].iloc[-tail_n:],
        df["value"].iloc[-tail_n:],
        color="black",
        lw=1.6,
        label="Фактичні дані",
    )
    plt.plot(
        future_dates,
        extrapolations[best_type][0],
        color="#d62728",
        lw=2.0,
        label=f"{best_type} + Composite Loss",
    )
    plt.plot(
        future_dates,
        ext_mse_mean,
        color="#ff7f0e",
        lw=2.0,
        ls="--",
        label=f"{best_type} + Standard MSE",
    )
    plt.axvline(df["date"].iloc[-1], color="gray", ls=":")
    plt.title(
        f"Ефект композитної функції втрат на довготривалій екстраполяції ({best_type})"
    )
    plt.xlabel("Дата")
    plt.ylabel("Значення (Value)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "09_loss_ablation_study.png"), dpi=200)
    plt.close()

    print(f"\n[DONE] Усі графіки та таблиця результатів збережені у папку '{cfg.output_dir}/'.")


if __name__ == "__main__":
    main()