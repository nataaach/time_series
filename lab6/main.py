from pathlib import Path
import pandas as pd
from src.data_loader import TimeSeriesLoader
from src.ts_profiler import TimeSeriesProfiler
from src.exp_smoothing import ExponentialSmoothingOptimizer
from src.regression_models import RegressionOptimizer
from src.kpi_evaluator import ModelEvaluator
from src.adaptive_selector import AdaptiveSelectorEngine
from src.visualizer import TimeSeriesVisualizer


def build_future_dates(dates: pd.DatetimeIndex, future_horizon: int) -> pd.DatetimeIndex:
    freq = pd.infer_freq(dates)
    if freq:
        return pd.date_range(start=dates[-1], periods=future_horizon + 1, freq=freq)[1:]
    diffs = dates.to_series().diff().dropna()
    step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)
    return pd.DatetimeIndex([dates[-1] + step * (i + 1) for i in range(future_horizon)])


def run_single_dataset_pipeline(
    ds,
    plots_dir: Path,
    results_dir: Path,
    test_ratio: float = 0.15,
    future_horizon: int = 12,
):
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    es_opt = ExponentialSmoothingOptimizer()
    reg_opt = RegressionOptimizer()
    vis = TimeSeriesVisualizer(output_dir=str(plots_dir))

    n = len(ds.values)
    test_len = max(8, int(round(n * test_ratio)))
    train_len = n - test_len
    train_y, test_y = ds.values[:train_len], ds.values[train_len:]

    print("=" * 90)
    print(f"ЛАБОРАТОРНА РОБОТА №6 | АНАЛІЗ ДАНИХ: {ds.name} ({ds.domain})")
    print("=" * 90)
    print(f"Розмір ряду: N={n} | Train={train_len} | Test={test_len} | Виявлений період сезонності={ds.seasonal_period}")

    # 1. Дослідження властивостей даних (EDA на train)
    profile = TimeSeriesProfiler.profile(train_y, seasonal_period=ds.seasonal_period)
    eda_plot = vis.plot_eda_and_decomposition(ds, train_len)

    print("\n[1] Властивості вхідного ряду:")
    print(
        f"    Mean={profile['mean']:.3f}, Std={profile['std']:.3f}, CV={profile['cv']:.3f}, "
        f"Min={profile['min']:.3f}, Max={profile['max']:.3f}\n"
        f"    Стаціонарність (ADF): {profile['is_stationary']} (stat={profile['adf_stat']:.4f}, p={profile['adf_pvalue']:.4f})\n"
        f"    Сила тренду F_T={profile['trend_strength']:.3f}, Сила сезонності F_S={profile['seasonality_strength']:.3f}\n"
        f"    R2_poly2={profile['r2_poly2']:.3f}, Індекс нелінійності={profile['nonlinearity_index']:.4f}, "
        f"Структурний злам={profile['structural_shift_index']:.3f}\n"
        f"    Викиди (MAD)={profile['outlier_ratio']*100:.2f}%, ACF(lag1)={profile['acf_lag1']:.3f}, ACF(lag2)={profile['acf_lag2']:.3f}"
    )

    # 2. Група вимог 1: Експоненціальне згладжування
    es_res = es_opt.fit_and_forecast(train_y, test_len, future_horizon, ds.seasonal_period)
    es_test_kpi = ModelEvaluator.regression_metrics(test_y, es_res["test_pred"], y_train_last=train_y[-1])

    print("\n[2] Група вимог 1 — Рейтинг моделей експоненціального згладжування:")
    print(pd.DataFrame(es_res["leaderboard"]).to_string(index=False))
    print(f"    -> Обрано: {es_res['best_model_name']} | Параметри: {es_res['params']}")

    # 3. Група вимог 2: Регресійні моделі
    reg_res = reg_opt.fit_and_forecast(train_y, test_len, future_horizon, ds.seasonal_period)
    reg_test_kpi = ModelEvaluator.regression_metrics(test_y, reg_res["test_pred"], y_train_last=train_y[-1])

    print("\n[3] Група вимог 2 — Рейтинг регресійних моделей:")
    print(pd.DataFrame(reg_res["leaderboard"]).to_string(index=False))
    print(f"    -> Обрано: {reg_res['best_model_name']} | Кількість параметрів: {reg_res['n_params']}")

    # 4. Група вимог 3: R&D порівняння та алгоритм вибору
    sel = AdaptiveSelectorEngine.recommend(profile, horizon=test_len)
    empirical_winner = "ExponentialSmoothing" if es_test_kpi["RMSE"] <= reg_test_kpi["RMSE"] else "Regression"
    selector_matched = (sel["primary_choice"] == empirical_winner)

    future_dates = build_future_dates(ds.dates, future_horizon)
    comp_plot = vis.plot_comparison_forecast(ds, train_len, es_res, reg_res, future_dates)

    summary_rows = []
    for paradigm, res, tkpi in [
        ("ExponentialSmoothing", es_res, es_test_kpi),
        ("Regression", reg_res, reg_test_kpi),
    ]:
        summary_rows.append({
            "Dataset": ds.name,
            "Paradigm": paradigm,
            "Best_Model": res["best_model_name"],
            "Train_RMSE": round(res["train_kpi"]["RMSE"], 4),
            "Train_R2": round(res["train_kpi"]["R2"], 4),
            "Test_MAE": round(tkpi["MAE"], 4),
            "Test_RMSE": round(tkpi["RMSE"], 4),
            "Test_MAPE_%": round(tkpi["MAPE_%"], 4),
            "Test_sMAPE_%": round(tkpi["sMAPE_%"], 4),
            "Test_R2": round(tkpi["R2"], 4),
            "Theil_U": round(tkpi["Theil_U"], 4),
            "Durbin_Watson": round(tkpi["Durbin_Watson"], 4),
            "AIC": round(res["aic"], 2),
            "Time_ms": round(res["fit_time_ms"], 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "kpi_comparison.csv", index=False)
    kpi_plot = vis.plot_kpi_benchmark(summary_df)

    selector_df = pd.DataFrame([{
        "Dataset": ds.name,
        "Score_ES": sel["score_es"],
        "Score_Reg": sel["score_reg"],
        "Confidence": sel["confidence"],
        "Selector_Choice": sel["recommended_paradigm"],
        "Empirical_Winner": empirical_winner,
        "Match": selector_matched,
    }])
    selector_df.to_csv(results_dir / "selector_verification.csv", index=False)

    print("\n[4] Група вимог 3 — Зведена таблиця KPI (Train та Test):")
    print(summary_df.to_string(index=False))

    print("\n[5] Робота алгоритму автоматичного вибору (AdaptiveSelectorEngine):")
    print(f"    Рекомендація: {sel['recommended_paradigm']} (S_ES={sel['score_es']}, S_Reg={sel['score_reg']}, Conf={sel['confidence']})")
    print(f"    Емпіричний переможець (Test RMSE): {empirical_winner} | Збіг із селектором: {selector_matched}")
    for r in sel["reasons"]:
        print(f"      - {r}")

    print("\nЗбережені графіки:")
    print(f"  1) EDA: {eda_plot}")
    print(f"  2) Порівняння прогнозів: {comp_plot}")
    print(f"  3) Порівняння KPI: {kpi_plot}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "lab2" / "data2.csv"
    PLOTS_DIR = BASE_DIR / "plots"
    RESULTS_DIR = BASE_DIR / "results"

    series = TimeSeriesLoader.load_data(
        file_path=str(DATA_PATH),
        date_col="Timestamp",
        value_col="Value",
    )

    run_single_dataset_pipeline(
        ds=series,
        plots_dir=PLOTS_DIR,
        results_dir=RESULTS_DIR,
        test_ratio=0.15,
        future_horizon=12,
    )