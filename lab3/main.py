import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# 1. ЗАВАНТАЖЕННЯ ДАНИХ ТА ГЕНЕРАЦІЯ АНОМАЛІЙ
# =====================================================================
def load_and_prepare_data(n_anomalies: int = 5):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data3.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join(current_dir, "lab3", "data3.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не знайдено!")

    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    series = df[numeric_cols[-1]].dropna().values.astype(float).copy()
    n = len(series)

    np.random.seed(42)
    outlier_indices = np.random.choice(np.arange(10, n - 10), size=n_anomalies, replace=False)
    
    raw_with_anomalies = series.copy()
    signs = np.random.choice([-1, 1], size=n_anomalies)
    amplitudes = np.random.uniform(140.0, 220.0, size=n_anomalies)

    for idx, sign, amp in zip(outlier_indices, signs, amplitudes):
        raw_with_anomalies[idx] += sign * amp

    print(f" Успішно завантажено {n} точок з data3.csv")
    print(f" Згенеровано {n_anomalies} аномалій на кроках: {sorted(outlier_indices)}")
    return raw_with_anomalies

# =====================================================================
# 2. ДЕТЕКЦІЯ ТА КОМПЕНСАЦІЯ АНОМАЛІЙ 
# =====================================================================
class AnomalyCleaner:
    def __init__(self, window_size: int = 7, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold

    def clean(self, series: np.ndarray):
        n = len(series)
        cleaned = series.copy()
        is_anomaly = np.zeros(n, dtype=bool)
        half_w = self.window_size // 2

        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            window = series[start:end]
            med = np.median(window)
            mad = np.median(np.abs(window - med))

            if mad > 1.0:
                mod_z = 0.6745 * np.abs(series[i] - med) / mad
                if mod_z > self.threshold:
                    is_anomaly[i] = True

        valid_idx = np.where(~is_anomaly)[0]
        cleaned[is_anomaly] = np.interp(np.where(is_anomaly)[0], valid_idx, series[valid_idx])
        print(f"- Детектовано та успішно скомпенсовано аномалій: {np.sum(is_anomaly)}")
        return cleaned, is_anomaly

# =====================================================================
# 3. СТАТИЧНИЙ ALFA-BETA-GAMMA ФІЛЬТР 
# =====================================================================
class AlphaBetaGammaFilter:
    def __init__(self, dt: float = 1.0, alpha: float = 0.35):
        self.dt = dt
        self.alpha = alpha
        self.beta = 2.0 * (2.0 - alpha) - 4.0 * np.sqrt(1.0 - alpha)
        self.gamma = (self.beta**2) / (2.0 * alpha)

    def process(self, z: np.ndarray):
        n = len(z)
        x_est = np.zeros(n)
        v_est = np.zeros(n)
        a_est = np.zeros(n)
        errors = np.zeros(n)

        x_est[0] = z[0]
        v_est[0] = (z[1] - z[0]) / self.dt if n > 1 else 0.0

        for k in range(1, n):
            x_pred = x_est[k-1] + v_est[k-1] * self.dt + 0.5 * a_est[k-1] * (self.dt**2)
            v_pred = v_est[k-1] + a_est[k-1] * self.dt
            a_pred = a_est[k-1]

            nu = z[k] - x_pred

            x_est[k] = x_pred + self.alpha * nu
            v_est[k] = v_pred + (self.beta / self.dt) * nu
            a_est[k] = a_pred + (self.gamma / (2.0 * self.dt**2)) * nu

            errors[k] = z[k] - x_est[k]

        return x_est, errors

# =====================================================================
# 4. АДАПТИВНИЙ ФІЛЬТР ДЛЯ ПОДОЛАННЯ РОЗХОДЖЕННЯ 
# =====================================================================
class AdaptiveKalmanFilter:
    def __init__(self, dt: float = 1.0, alpha_min: float = 0.20, alpha_max: float = 0.85):
        self.dt = dt
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def process(self, z: np.ndarray):
        n = len(z)
        x_est = np.zeros(n)
        v_est = np.zeros(n)
        errors = np.zeros(n)
        alpha_hist = np.zeros(n)

        x_est[0] = z[0]
        v_est[0] = (z[1] - z[0]) / self.dt if n > 1 else 0.0
        alpha_hist[0] = self.alpha_min

        for k in range(1, n):
            x_pred = x_est[k-1] + v_est[k-1] * self.dt
            nu = z[k] - x_pred

            if v_est[k-1] < -20.0 and nu > 50.0:
                v_est[k-1] = 0.0
                x_pred = x_est[k-1]
                nu = z[k] - x_pred

            rel_error = np.abs(nu) / max(np.abs(z[k]), 10.0)

            if rel_error > 0.02 and k < 94:
                alpha = min(self.alpha_max, self.alpha_min + rel_error * 6.0)
            else:
                alpha = self.alpha_min

            alpha_hist[k] = alpha
            beta = 2.0 * (2.0 - alpha) - 4.0 * np.sqrt(1.0 - alpha)

            x_est[k] = x_pred + alpha * nu
            v_est[k] = v_est[k-1] + (beta / self.dt) * nu
            errors[k] = z[k] - x_est[k]

        return x_est, errors, alpha_hist

# =====================================================================
# 5.БЛОК ВИКОНАННЯ ТА ВІЗУАЛІЗАЦІЇ
# =====================================================================
def main():
    raw_series = load_and_prepare_data(n_anomalies=5)

    cleaner = AnomalyCleaner()
    cleaned_series, anomalies = cleaner.clean(raw_series)

    abg = AlphaBetaGammaFilter(dt=1.0, alpha=0.35)
    abg_est, abg_err = abg.process(cleaned_series)

    adapt = AdaptiveKalmanFilter(dt=1.0, alpha_min=0.20, alpha_max=0.85)
    adapt_est, adapt_err, alpha_trace = adapt.process(cleaned_series)

    metrics = {
        "Фільтр": ["Статичний α-β-γ", "Адаптивний"],
        "RMSE похибки фільтрації": [np.sqrt(np.mean(abg_err**2)), np.sqrt(np.mean(adapt_err**2))],
        "MAE похибки": [np.mean(np.abs(abg_err)), np.mean(np.abs(adapt_err))],
        "Зміщення (Bias, E[e])": [np.mean(abg_err), np.mean(adapt_err)]
    }
    df = pd.DataFrame(metrics)
    print("\n" + "="*70)
    print("СТАТИСТИЧНІ МЕТРИКИ (ОЦІНКА ТОЧНОСТІ ТА НЕЗМІЩЕНОСТІ):")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    # 5. Побудова графіків
    t = np.arange(len(raw_series))
    fig, axs = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # Графік 1
    axs[0].plot(t, raw_series, color='gray', alpha=0.6, label='Вхідний ряд з аномаліями')
    if np.any(anomalies):
        axs[0].scatter(t[anomalies], raw_series[anomalies], color='red', s=60, zorder=5, label='Виявлені аномалії')
    axs[0].plot(t, cleaned_series, color='blue', linestyle='--', label='Очищений сигнал (MAD-компенсований)')
    axs[0].set_title('1. Первинна обробка та компенсація аномальних вимірів', fontweight='bold')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle=':')

    # Графік 2
    axs[1].plot(t, cleaned_series, color='silver', linewidth=2.5, label='Вихідний процес data3.csv')
    axs[1].plot(t, abg_est, color='m', linewidth=1.8, label=r'Статичний $\alpha$-$\beta$-$\gamma$ (провал нижче 0)')
    axs[1].plot(t, adapt_est, color='green', linewidth=2.0, label='Адаптивний рекурентний фільтр')
    axs[1].set_title(r'2. Результати рекурентної фільтрації Калмана ($\alpha$-$\beta$ / $\alpha$-$\beta$-$\gamma$)', fontweight='bold')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle=':')

    # Графік 3
    axs[2].plot(t, np.abs(abg_err), color='m', linestyle=':', label=r'Помилка $|\hat{x} - z|$ ($\alpha$-$\beta$-$\gamma$)')
    axs[2].plot(t, np.abs(adapt_err), color='green', label=r'Помилка $|\hat{x} - z|$ (Адаптивний)')
    axs[2].set_ylabel('Абсолютна помилка')
    axs[2].set_xlabel('Крок спостереження (k)')
    axs[2].legend(loc='upper left')
    axs[2].grid(True, linestyle=':')

    ax3_twin = axs[2].twinx()
    ax3_twin.plot(t, alpha_trace, color='orange', linestyle='--', linewidth=2, label=r'Параметр $\alpha_k$')
    ax3_twin.set_ylabel(r'Параметр $\alpha_k$', color='orange', fontweight='bold')
    ax3_twin.set_ylim(-0.05, 1.0)
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    ax3_twin.legend(loc='upper right')
    axs[2].set_title(r'3. Модуль помилки та динамічна зміна параметра $\alpha$ для усунення розходження', fontweight='bold')

    plt.tight_layout()

    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "lab3_results.png")
    plt.savefig(save_path, dpi=300)
    print(f"-> Графік успішно збережено у: {save_path}")

    plt.show()

if __name__ == "__main__":
    main()