import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mt

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ==========================================
def load_data(csv_path="data2.csv"):
    df = pd.read_csv(csv_path)
    t = df["t"].values
    y_original = df["Value"].values
    return t, y_original

# ==========================================
# 2. ГЕНЕРАЦІЯ ТА ОЧИЩЕННЯ АНОМАЛІЙ (ЗБІЛЬШЕНА КІЛЬКІСТЬ)
# ==========================================
def inject_anomalies(y):
    """Штучне внесення більшої кількості аномалій"""
    np.random.seed(42)
    y_corrupted = y.copy()
    anomaly_indices = [5, 15, 30, 42, 55, 68, 75, 88, 95]
    
    noise_level = np.std(y) * 6  
    
    for idx in anomaly_indices:
        sign = np.random.choice([-1, 1])
        y_corrupted[idx] += sign * noise_level * np.random.uniform(1.5, 3.5)
        
    return y_corrupted, anomaly_indices

def detect_and_clean_anomalies_mad(series, window=7, threshold=3.0):
    """Робастний алгоритм очищення на базі медіанного відхилення (MAD)"""
    s = pd.Series(series)
    
    rolling_median = s.rolling(window=window, center=True, min_periods=3).median()
    diff = np.abs(s - rolling_median)
    rolling_mad = diff.rolling(window=window, center=True, min_periods=3).median()
    
    scale = np.where(rolling_mad > 1e-4, rolling_mad * 1.4826, 1e-6)

    z_scores = diff / scale
    outlier_mask = z_scores > threshold

    s_cleaned = s.copy()
    s_cleaned[outlier_mask] = np.nan
    s_cleaned = s_cleaned.interpolate(method='linear').bfill().ffill()
    
    return s_cleaned.to_numpy(), outlier_mask.to_numpy()

# ==========================================
# 3. R&D: АЛГОРИТМ НАУ (ОПТИМАЛЬНИЙ СТУПІНЬ ЗА ПОХІДНИМИ)
# ==========================================
def degree_polynomial_derivatives_scor(SL: np.ndarray, dt: float) -> int:
    """Функція визначення оптимального ступеня полінома (на основі похідних з методички)"""
    iter = len(SL)
    y1, y2, y3, y4, y5 = np.zeros(iter), np.zeros(iter), np.zeros(iter), np.zeros(iter), np.zeros(iter)

    for i in range(iter):
        if i < (iter - 1): y1[i] = (SL[i+1] - SL[i]) / dt
        if i < (iter - 2): y2[i] = (SL[i+2] - 2 * SL[i+1] + SL[i]) / (dt ** 2)
        if i < (iter - 3): y3[i] = (SL[i+3] - 3 * SL[i+2] + 3 * SL[i+1] - SL[i]) / (dt ** 3)
        if i < (iter - 4): y4[i] = (SL[i+4] - 4 * SL[i+3] + 6 * SL[i+2] - 4 * SL[i+1] + SL[i]) / (dt ** 4)
        if i < (iter - 5): y5[i] = (SL[i+5] - 5 * SL[i+4] + 10 * SL[i+3] - 10 * SL[i+2] + 5 * SL[i+1] - SL[i]) / (dt ** 5)

    scv_y1_exp = mt.sqrt(np.var(y1))
    scv_y2_exp = mt.sqrt(np.var(y2))
    scv_y3_exp = mt.sqrt(np.var(y3))
    scv_y4_exp = mt.sqrt(np.var(y4))
    scv_y5_exp = mt.sqrt(np.var(y5))

    var_SL = np.var(SL)
    dt = max(dt, 1.0) 
    
    scv_y1_theor = mt.sqrt((2 * var_SL) / (dt ** 2))
    scv_y2_theor = mt.sqrt((2 * var_SL) / (dt ** 4))
    scv_y3_theor = mt.sqrt((2 * var_SL) / (dt ** 6))
    scv_y4_theor = mt.sqrt((2 * var_SL) / (dt ** 8))
    scv_y5_theor = mt.sqrt((2 * var_SL) / (dt ** 10))

    dalta_array = np.zeros(5)
    dalta_array[0] = abs(scv_y1_exp - scv_y1_theor)
    dalta_array[1] = abs(scv_y2_exp - scv_y2_theor)
    dalta_array[2] = abs(scv_y3_exp - scv_y3_theor)
    dalta_array[3] = abs(scv_y4_exp - scv_y4_theor)
    dalta_array[4] = abs(scv_y5_exp - scv_y5_theor)

    m_polinom = np.argmin(dalta_array) + 1
    
    print(f"\n--- Робота R&D алгоритму (на основі похідних) ---")
    print(f"Масив дельт (різниця СКВ): {dalta_array}")
    return m_polinom

# ==========================================
# 4. МНК + РОЗРАХУНОК AIC / RMSE
# ==========================================
def custom_lsm_fit_predict(x, y, degree, x_pred):
    """Кастомна реалізація МНК з розрахунком R^2, RMSE та AIC"""
    X = np.vander(x, degree + 1)
    
    X_T = X.T
    coeffs = np.linalg.inv(X_T @ X) @ X_T @ y

    X_pred_matrix = np.vander(x_pred, degree + 1)
    y_pred = X_pred_matrix @ coeffs

    y_fit = X @ coeffs
    
    n = len(y)
    k = degree + 1 
    
    rss = np.sum((y - y_fit) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    
    r2 = 1 - (rss / tss)
    rmse = np.sqrt(rss / n)
    aic = n * np.log(rss / n) + 2 * k
    
    return coeffs, y_fit, y_pred, r2, rmse, aic

# ==========================================
# 5. ALPHA-BETA ФІЛЬТР
# ==========================================
def alpha_beta_filter(data, dt=1.0, alpha=0.4, beta=0.01):
    n = len(data)
    pos_est = np.zeros(n)
    vel_est = np.zeros(n)
    
    pos_est[0] = data[0]
    vel_est[0] = (data[1] - data[0]) / dt
    
    for k in range(1, n):
        pos_pred = pos_est[k - 1] + vel_est[k - 1] * dt
        vel_pred = vel_est[k - 1]
        
        residual = data[k] - pos_pred

        pos_est[k] = pos_pred + alpha * residual
        vel_est[k] = vel_pred + (beta / dt) * residual
        
    return pos_est, vel_est

def main():
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    print("=" * 60)
    print(" ЛАБОРАТОРНА РОБОТА 2: СТАТИСТИЧНЕ НАВЧАННЯ")
    print("=" * 60)

    # 1. Завантаження
    t, y_original = load_data()
    n_samples = len(t)
    print(f"\n[1/6] Дані успішно завантажено. Кількість відліків: {n_samples}")

    # 2. Аномалії
    y_noisy, true_anomalies = inject_anomalies(y_original)
    y_cleaned, detected_mask = detect_and_clean_anomalies_mad(y_noisy)
    detected_indices = np.where(detected_mask)[0]
    print(f"\n[2/6] Штучно внесено аномалії (індекси): {true_anomalies}")
    print(f"      Детектовано та очищено (MAD): {detected_indices.tolist()}")

    # 3. R&D: Визначення оптимального ступеня за похідними
    dt = 1.0 
    optimal_degree_rd = degree_polynomial_derivatives_scor(y_cleaned, dt)
    if optimal_degree_rd < 2: 
        optimal_degree_rd = 2 

    # 4. Класичний критерій AIC та таблиця
    n_forecast = int(0.5 * n_samples)
    t_forecast = np.arange(n_samples, n_samples + n_forecast)
    
    print("\n[3/6] Порівняльна таблиця (Класичний критерій AIC)")
    print(f"{'Ступінь':<10} | {'R^2':<10} | {'RMSE':<10} | {'AIC':<10}")
    print("-" * 50)
    
    best_aic = float('inf')
    best_degree_aic = 1
    
    for deg in range(1, 6):
        _, _, _, r2_tmp, rmse_tmp, aic_tmp = custom_lsm_fit_predict(t, y_cleaned, deg, t_forecast)
        print(f"{deg:<10} | {r2_tmp:<10.4f} | {rmse_tmp:<10.4f} | {aic_tmp:<10.2f}")
        if aic_tmp < best_aic:
            best_aic = aic_tmp
            best_degree_aic = deg
            
    print("-" * 50)
    print(f"Оптимум за класичним AIC: ступінь {best_degree_aic}")
    print(f"Оптимум за R&D методом (похідні): ступінь {optimal_degree_rd}")

    # Для подальших розрахунків МНК та прогнозу використовуємо ступінь за AIC (або можете замінити на optimal_degree_rd)
    chosen_degree = best_degree_aic 

    # 5. МНК + Прогноз
    coeffs, y_fit, y_pred_future, r2, rmse, aic = custom_lsm_fit_predict(t, y_cleaned, chosen_degree, t_forecast)
    print(f"\n[4/6] МНК виконано. Обраний ступінь: {chosen_degree}, R^2: {r2:.4f}, AIC: {aic:.2f}")
    print(f"      Прогноз на {n_forecast} кроків вперед успішно згенеровано.")

    ab_pos, ab_vel = alpha_beta_filter(y_cleaned) 
    print(f"\n[5/6] Рекурентне згладжування (Alpha-Beta) виконано.")
    print(f"      Поточна оцінена швидкість приросту: {ab_vel[-1]:.4f} од/час")

    # 6. ВІЗУАЛІЗАЦІЯ
    print("\n[6/6] Збереження графіків у папку plots/ ...")
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, y_noisy, color="gray", linestyle=":", label="Ряд з аномаліями")
    plt.plot(t, y_cleaned, color="#1b5d35", linewidth=2, label="Очищений ряд")
    plt.scatter(t[detected_mask], y_noisy[detected_mask], color="red", zorder=5, label="Знайдені аномалії")
    plt.title("Виявлення та очищення аномалій (MAD Z-Score) з підвищеним шумом")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "1_anomaly_cleaning.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(t, y_cleaned, color="black", alpha=0.5, label="Реальні дані (очищені)")
    plt.plot(t, y_fit, color="red", linewidth=2, label=f"МНК тренд (Ступінь {chosen_degree})")
    plt.plot(t_forecast, y_pred_future, color="magenta", linewidth=2, linestyle="--", label="Екстраполяція (+50%)")
    plt.plot(t, ab_pos, color="green", linestyle="-.", linewidth=2, label="Alpha-Beta фільтрація")
    plt.axvline(n_samples - 1, color="gray", linestyle="--", label="Межа спостереження")
    plt.title(f"Статистичне навчання МНК (Ступінь={chosen_degree}, R²={r2:.4f}) та Екстраполяція")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "2_lsm_forecast_alphabeta.png"), dpi=300)
    plt.close()

    print("\nГрафіки успішно збережено у папку plots/")

if __name__ == "__main__":
    main()