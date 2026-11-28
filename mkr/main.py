import os
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

warnings.filterwarnings("ignore")

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_real_time_series():
    url = (
        "https://bank.gov.ua/NBU_Exchange/exchange_site?"
        "start=20230101&end=20250201&valcode=usd&sort=exchangedate&order=asc&json"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["exchangedate"], format="%d.%m.%Y")
    df["value"] = df["rate"].astype(float)
    df = df[["date", "value"]].sort_values("date").drop_duplicates("date").reset_index(drop=True)

    df["t_days"] = (df["date"] - df["date"].iloc[0]).dt.days.astype(float)

    series_name = "Офіційний курс USD/UAH (НБУ)"
    unit = "грн за 1 USD"
    period_days = 365.25
    print(f"[INFO] Дані успішно завантажено з API НБУ: {len(df)} записів.")
    return df, series_name, unit, period_days


def build_design_matrix(t_norm: np.ndarray, model_type: str, period_norm: float) -> np.ndarray:

    ones = np.ones_like(t_norm)
    if model_type == "linear":
        return np.column_stack([ones, t_norm])
    elif model_type == "poly2":
        return np.column_stack([ones, t_norm, t_norm ** 2])
    elif model_type == "poly3":
        return np.column_stack([ones, t_norm, t_norm ** 2, t_norm ** 3])
    elif model_type == "trend_harmonic":
        omega = 2.0 * np.pi / period_norm
        return np.column_stack([
            ones,
            t_norm,
            t_norm ** 2,
            np.sin(omega * t_norm),
            np.cos(omega * t_norm),
        ])
    else:
        raise ValueError(f"Невідомий тип моделі: {model_type}")


def fit_ols_model(X_train: np.ndarray, y_train: np.ndarray):
    return sm.OLS(y_train, X_train).fit()


def predict_with_ci(ols_res, X_train: np.ndarray, X_target: np.ndarray, alpha: float = 0.05):
    theta = ols_res.params
    y_pred = X_target @ theta

    n, p = X_train.shape
    df_resid = n - p
    residuals = ols_res.resid
    s2 = np.sum(residuals ** 2) / df_resid
    s = np.sqrt(s2)

    XtX_inv = np.linalg.pinv(X_train.T @ X_train)
    leverage = np.einsum("ij,jk,ik->i", X_target, XtX_inv, X_target)
    se_pred = s * np.sqrt(1.0 + leverage)

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df_resid)
    lower = y_pred - t_crit * se_pred
    upper = y_pred + t_crit * se_pred
    return y_pred, lower, upper


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    mape = np.mean(np.abs(err / y_true)) * 100.0
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def main():
    df, series_name, unit, period_days = load_real_time_series()

    n_total = len(df)
    split_idx = int(n_total * 0.85)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print("\n" + "=" * 70)
    print(f"ДОСЛІДЖУВАНИЙ РЯД: {series_name}")
    print(f"Період: {df['date'].iloc[0].strftime('%d.%m.%Y')} — {df['date'].iloc[-1].strftime('%d.%m.%Y')}")
    print(f"Всього точок: {n_total} | Train (85%): {len(df_train)} | Test (15%): {len(df_test)}")
    print(f"Пропуски значень: {df['value'].isna().sum()}")
    print("=" * 70)

    t_scale = df_train["t_days"].iloc[-1] if df_train["t_days"].iloc[-1] > 0 else 1.0
    t_train_norm = df_train["t_days"].values / t_scale
    t_test_norm = df_test["t_days"].values / t_scale
    period_norm = period_days / t_scale

    y_train = df_train["value"].values
    y_test = df_test["value"].values

    plt.figure()
    plt.plot(df_train["date"], y_train, label="Навчальна вибірка (інтервал спостереження)", color="#1f77b4", lw=1.8)
    plt.plot(df_test["date"], y_test, label="Тестова вибірка (перевірка екстраполяції)", color="#ff7f0e", lw=1.8)
    plt.axvline(df_test["date"].iloc[0], color="red", linestyle="--", alpha=0.7, label="Межа інтервалу спостереження")
    plt.title(f"Вихідний часовий ряд: {series_name}")
    plt.xlabel("Дата")
    plt.ylabel(unit)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_raw_series.png"), dpi=200)
    plt.close()

    models_spec = [
        ("linear", "Лінійна (ст. 1)"),
        ("poly2", "Квадратична (ст. 2)"),
        ("poly3", "Кубічна (ст. 3)"),
        ("trend_harmonic", "Тренд (ст. 2) + гармоніки"),
    ]

    results_table = []
    fitted_objects = {}

    plt.figure()
    plt.plot(df["date"], df["value"], label="Фактичні дані", color="black", alpha=0.55, lw=1.5)
    plt.axvline(df_test["date"].iloc[0], color="red", linestyle="--", alpha=0.7)

    colors = ["#2ca02c", "#d62728", "#9467bd", "#17becf"]

    for (m_key, m_label), color in zip(models_spec, colors):
        X_tr = build_design_matrix(t_train_norm, m_key, period_norm)
        X_te = build_design_matrix(t_test_norm, m_key, period_norm)

        ols_res = fit_ols_model(X_tr, y_train)
        y_tr_pred, _, _ = predict_with_ci(ols_res, X_tr, X_tr)
        y_te_pred, _, _ = predict_with_ci(ols_res, X_tr, X_te)

        m_train = evaluate_metrics(y_train, y_tr_pred)
        m_test = evaluate_metrics(y_test, y_te_pred)

        results_table.append({
            "key": m_key,
            "Модель": m_label,
            "p": X_tr.shape[1],
            "R2_train": m_train["R2"],
            "AIC_train": ols_res.aic,
            "RMSE_train": m_train["RMSE"],
            "RMSE_test": m_test["RMSE"],
            "MAE_test": m_test["MAE"],
            "MAPE_test_%": m_test["MAPE"],
        })
        fitted_objects[m_key] = (ols_res, X_tr, y_tr_pred, y_te_pred)

        full_pred = np.concatenate([y_tr_pred, y_te_pred])
        plt.plot(df["date"], full_pred, label=f"{m_label} (Test RMSE={m_test['RMSE']:.3f})", color=color, lw=2.0)

    plt.title(f"Порівняння моделей МНК на навчальній та тестовій ділянках ({series_name})")
    plt.xlabel("Дата")
    plt.ylabel(unit)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_models_comparison.png"), dpi=200)
    plt.close()

    res_df = pd.DataFrame(results_table)
    print("\nПОРІВНЯЛЬНА ТАБЛИЦЯ МОДЕЛЕЙ МНК:")
    print(res_df.drop(columns=["key"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_row = res_df.loc[res_df["RMSE_test"].idxmin()]
    best_key = best_row["key"]
    best_name = best_row["Модель"]
    best_ols, _, _, _ = fitted_objects[best_key]

    print("\n" + "-" * 70)
    print(f"НАЙКРАЩА МОДЕЛЬ ЗА ПОЗАВИБІРКОВОЮ ТОЧНІСТЮ: {best_name}")
    print(f"Параметри theta_hat: {np.round(best_ols.params, 6)}")
    print(f"F-статистика Фішера: F = {best_ols.fvalue:.4f}, p-value = {best_ols.f_pvalue:.4e}")

    residuals = best_ols.resid
    dw_stat = durbin_watson(residuals)
    shapiro_sample = residuals if len(residuals) <= 5000 else np.random.choice(residuals, 5000, replace=False)
    sw_stat, sw_p = stats.shapiro(shapiro_sample)

    print(f"Середнє залишків: {np.mean(residuals):.6f} | Std залишків: {np.std(residuals, ddof=1):.4f}")
    print(f"Статистика Дурбіна-Вотсона (DW): {dw_stat:.4f}")
    print(f"Тест Шапіро-Вілка: W = {sw_stat:.4f}, p-value = {sw_p:.4e}")
    print("-" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(df_train["date"], residuals, color="#1f77b4", lw=1.2)
    axes[0, 0].axhline(0, color="red", linestyle="--", lw=1.2)
    axes[0, 0].set_title("Залишки моделі на інтервалі спостереження")
    axes[0, 0].set_xlabel("Дата")
    axes[0, 0].set_ylabel("Залишок")

    axes[0, 1].hist(residuals, bins=25, density=True, color="#2ca02c", alpha=0.65, edgecolor="black")
    x_grid = np.linspace(residuals.min(), residuals.max(), 200)
    axes[0, 1].plot(x_grid, stats.norm.pdf(x_grid, np.mean(residuals), np.std(residuals)), "r-", lw=2)
    axes[0, 1].set_title("Гістограма розподілу залишків")
    axes[0, 1].set_xlabel("Значення залишку")

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q графік залишків (перевірка нормальності)")

    plot_acf(residuals, ax=axes[1, 1], lags=min(30, len(residuals) // 3), title="Автокореляційна функція залишків (ACF)")

    fig.suptitle(f"Діагностика адекватності найкращої моделі ({best_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_residuals_diagnostics.png"), dpi=200)
    plt.close()

    horizon_days = 60
    t_full_scale = df["t_days"].iloc[-1]
    t_full_norm = df["t_days"].values / t_full_scale
    period_full_norm = period_days / t_full_scale

    X_full = build_design_matrix(t_full_norm, best_key, period_full_norm)
    y_full = df["value"].values
    final_ols = fit_ols_model(X_full, y_full)

    future_days_offset = np.arange(1, horizon_days + 1, dtype=float)
    future_t_days = t_full_scale + future_days_offset
    future_t_norm = future_t_days / t_full_scale
    future_dates = pd.date_range(start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    X_future = build_design_matrix(future_t_norm, best_key, period_full_norm)

    y_fit_full, fit_low, fit_up = predict_with_ci(final_ols, X_full, X_full)
    y_fut_pred, fut_low, fut_up = predict_with_ci(final_ols, X_full, X_future)

    print(f"\nПРОГНОЗ ПОЗА МЕЖАМИ ІНТЕРВАЛУ СПОСТЕРЕЖЕННЯ (горизонт = {horizon_days} днів):")
    for step in [1, 30, 60]:
        idx = step - 1
        print(
            f"  +{step:2d} дн. ({future_dates[idx].strftime('%d.%m.%Y')}): "
            f"прогноз = {y_fut_pred[idx]:.4f} | 95% ДІ: [{fut_low[idx]:.4f}; {fut_up[idx]:.4f}]"
        )

    plt.figure()
    plt.plot(df["date"], y_full, label="Фактичні спостереження", color="#1f77b4", lw=1.6)
    plt.plot(df["date"], y_fit_full, label=f"МНК-модель ({best_name})", color="#2ca02c", lw=2.0)
    plt.plot(future_dates, y_fut_pred, label=f"Прогноз на {horizon_days} днів", color="#d62728", lw=2.2)
    plt.fill_between(future_dates, fut_low, fut_up, color="#d62728", alpha=0.2, label="95% довірчий інтервал прогнозу")
    plt.axvline(df["date"].iloc[-1], color="black", linestyle="--", alpha=0.7, label="Кінець інтервалу спостереження")
    plt.title(f"Екстраполяція часового ряду методом найменших квадратів ({series_name})")
    plt.xlabel("Дата")
    plt.ylabel(unit)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_future_forecast.png"), dpi=200)
    plt.close()

    print(f"\n Всі графіки збережено у папку '{PLOTS_DIR}/'.")


if __name__ == "__main__":
    main()