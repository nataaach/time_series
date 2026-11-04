import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ts_analyzer import TSAnalyzer
from ts_synthesizer import TSSynthesizer

CSV_DATA = """Order ID,Order Date,Customer ID,Customer Name,Address,City,State,ZIP/Postal Code,Country/Region,Salesperson,Region,Shipped Date,Shipper Name,Ship Name,Ship Address,Ship City,Ship State,Ship ZIP/Postal Code,Ship Country/Region,Payment Type,Product Name,Category,Unit Price,Quantity,Revenue,Shipping Fee
1433,01/27/15,27,Company AA,789 27th Street,Las Vegas,NV,99999,USA,Mariya Sergienko,West,01/29/15,Shipping Company B,Karen Toh,789 27th Street,Las Vegas,NV,99999,USA,Check,Beer,Beverages,14.00,49,686.00,66.54
1434,01/27/15,27,Company AA,789 27th Street,Las Vegas,NV,99999,USA,Mariya Sergienko,West,01/29/15,Shipping Company B,Karen Toh,789 27th Street,Las Vegas,NV,99999,USA,Check,Dried Plums,Dried Fruit & Nuts,3.50,47,164.50,16.61
1435,01/04/15,4,Company D,123 4th Street,New York,NY,99999,USA,Andrew Cencini,East,01/06/15,Shipping Company A,Christina Lee,123 4th Street,New York,NY,99999,USA,Credit Card,Dried Pears,Dried Fruit & Nuts,30.00,69,2070.00,198.72
1436,01/04/15,4,Company D,123 4th Street,New York,NY,99999,USA,Andrew Cencini,East,01/06/15,Shipping Company A,Christina Lee,123 4th Street,New York,NY,99999,USA,Credit Card,Dried Apples,Dried Fruit & Nuts,53.00,89,4717.00,448.12
1437,01/04/15,4,Company D,123 4th Street,New York,NY,99999,USA,Andrew Cencini,East,01/06/15,Shipping Company A,Christina Lee,123 4th Street,New York,NY,99999,USA,Credit Card,Dried Plums,Dried Fruit & Nuts,3.50,11,38.50,3.73
1438,01/12/15,12,Company L,123 12th Street,Las Vegas,NV,99999,USA,Mariya Sergienko,West,01/14/15,Shipping Company B,John Edwards,123 12th Street,Las Vegas,NV,99999,USA,Credit Card,Chai,Beverages,18.00,81,1458.00,141.43
1439,01/12/15,12,Company L,123 12th Street,Las Vegas,NV,99999,USA,Mariya Sergienko,West,01/14/15,Shipping Company B,John Edwards,123 12th Street,Las Vegas,NV,99999,USA,Credit Card,Coffee,Beverages,46.00,44,2024.00,198.35
1440,01/08/15,8,Company H,123 8th Street,Portland,OR,99999,USA,Nancy Freehafer,North,01/10/15,Shipping Company C,Elizabeth Andersen,123 8th Street,Portland,OR,99999,USA,Credit Card,Chocolate Biscuits Mix,Baked Goods & Mixes,9.20,38,349.60,36.01
1441,01/04/15,4,Company D,123 4th Street,New York,NY,99999,USA,Andrew Cencini,East,01/06/15,Shipping Company C,Christina Lee,123 4th Street,New York,NY,99999,USA,Check,Chocolate Biscuits Mix,Baked Goods & Mixes,9.20,88,809.60,79.34
1442,01/29/15,29,Company CC,789 29th Street,Denver,CO,99999,USA,Jan Kotas,West,01/31/15,Shipping Company B,Soo Jung Lee,789 29th Street,Denver,CO,99999,USA,Check,Chocolate,Candy,12.75,94,1198.50,122.25
1443,01/03/15,3,Company C,123 3rd Street,Los Angelas,CA,99999,USA,Mariya Sergienko,West,01/05/15,Shipping Company B,Thomas Axerr,123 3rd Street,Los Angelas,CA,99999,USA,Cash,Clam Chowder,Soups,9.65,91,878.15,92.21
1444,01/06/15,6,Company F,123 6th Street,Milwaukee,WI,99999,USA,Michael Neipper,North,01/08/15,Shipping Company B,Francisco Pérez-Olaeta,123 6th Street,Milwaukee,WI,99999,USA,Credit Card,Curry Sauce,Sauces,40.00,32,1280.00,133.12
1445,01/28/15,28,Company BB,789 28th Street,Memphis,TN,99999,USA,Anne Larsen,South,01/30/15,Shipping Company C,Amritansh Raghav,789 28th Street,Memphis,TN,99999,USA,Check,Coffee,Beverages,46.00,55,2530.00,253.00
1446,01/08/15,8,Company H,123 8th Street,Portland,OR,99999,USA,Nancy Freehafer,North,01/10/15,Shipping Company C,Elizabeth Andersen,123 8th Street,Portland,OR,99999,USA,Check,Chocolate,Candy,12.75,47,599.25,61.72
1447,01/10/15,10,Company J,123 10th Street,Chicago,IL,99999,USA,Laura Giussani,East,01/12/15,Shipping Company B,Roland Wacker,123 10th Street,Chicago,IL,99999,USA,Credit Card,Green Tea,Beverages,2.99,90,269.10,27.72
"""

def main():
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 80)
    print("  ЛАБОРАТОРНА РОБОТА №4: ПОШУК ЗАКОНОМІРНОСТЕЙ В СТРУКТУРАХ TIME SERIES")
    print("=" * 80)

    df = pd.read_csv(io.StringIO(CSV_DATA))
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed')

    daily_ts = df.groupby('Order Date').agg({
        'Revenue': 'sum',
        'Quantity': 'sum',
        'Shipping Fee': 'sum'
    }).asfreq('D', fill_value=0.0)

    print(f"\n[1/5] Завантаження та щоденна агрегація даних виконана успішно.")
    print(f"Кількість діб спостереження: {len(daily_ts)} відліків.")

    analyzer = TSAnalyzer(daily_ts['Revenue'])
    props = analyzer.calculate_properties()

    print("\n--- 1. Фізичні та статистичні властивості ряду Revenue ---")
    print(f"  Середнє значення (Mean)      : {props['mean']:.2f} $")
    print(f"  Стандартне відхилення (STD)  : {props['std']:.2f} $")
    print(f"  Коефіцієнт асиметрії (Skew)  : {props['skewness']:.4f}")
    print(f"  Коефіцієнт ексцесу (Kurtosis): {props['kurtosis']:.4f}")
    print(f"  ADF статистика               : {props['adf_stat']:.4f}")
    print(f"  ADF p-value                  : {props['adf_pvalue']:.5f}")
    print(f"  Стаціонарність ряду          : {'СТАЦІОНАРНИЙ' if props['is_stationary'] else 'НЕСТАЦІОНАРНИЙ'}")

    decomp = analyzer.decompose(period=7)
    trend = decomp.trend
    seasonal = decomp.seasonal
    resid = decomp.resid.dropna()

    print("\n--- 2. Результати адитивної декомпозиції ---")
    print(f"  Середня амплітуда сезонності (Peak-to-Peak) : {np.ptp(seasonal):.2f} $")
    print(f"  Дисперсія шуму залишків (Residual Variance)  : {np.var(resid):.2f} $^2")

    corr_matrix = TSAnalyzer.compute_correlation_matrix(daily_ts)
    print("\n--- 3. Багатофакторна кореляційна матриця ---")
    print(corr_matrix.round(4))

    category_pivot = df.pivot_table(index='Order Date', columns='Category', values='Revenue', aggfunc='sum')
    category_pivot = category_pivot.asfreq('D', fill_value=0.0)
    clusters = TSAnalyzer.cluster_categories(category_pivot, n_clusters=2)

    print("\n--- 4. Кластеризація профілів динаміки попиту (K-Means) ---")
    cluster_desc = {
        0: "Високочастотний стабільний попит",
        1: "Імпульсний B2B попит великих партій"
    }
    for cat, cl in clusters.items():
        print(f"  Категорія '{cat:<22}' --> Кластер {cl} ({cluster_desc[cl]})")

    t = np.arange(len(daily_ts))
    poly = np.polyfit(t, daily_ts['Revenue'].values, 1)

    synthesizer = TSSynthesizer(
        length=len(daily_ts),
        slope=poly[0],
        intercept=poly[1],
        period=7,
        amplitude=float(np.std(seasonal)),
        noise_std=float(np.std(resid))
    )
    synthetic_series = synthesizer.generate()
    adequacy = TSSynthesizer.verify_adequacy(daily_ts['Revenue'], synthetic_series)

    print("\n--- 5. Моделювання та верифікація адекватності (KS-test) ---")
    print(f"  Параметри синтезу: Slope={poly[0]:.2f}, Intercept={poly[1]:.2f}, "
          f"Amp={np.std(seasonal):.2f}, Noise_STD={np.std(resid):.2f}")
    print(f"  Критерій Колмогорова-Смірнова (KS Statistic) : {adequacy['ks_stat']:.4f}")
    print(f"  KS p-value                                   : {adequacy['ks_pvalue']:.4f}")
    print(f"  Відносна похибка математичного сподівання    : {adequacy['mean_rel_error'] * 100:.2f} %")
    print(f"  Відносна похибка дисперсії                   : {adequacy['std_rel_error'] * 100:.2f} %")
    status = "АДЕКВАТНОЮ (p-value >= 0.05, розподіли ідентичні)" if adequacy['is_adequate'] else "НЕАДЕКВАТНОЮ"
    print(f"  ВИСНОВОК ВЕРИФІКАЦІЇ: МОДЕЛЬ Є {status}")

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(daily_ts.index, daily_ts['Revenue'], color='#1f77b4', lw=2)
    axes[0].set_title("1. Observed Time Series (Daily Revenue)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(daily_ts.index, trend, color='#ff7f0e', lw=2)
    axes[1].set_title("2. Extracted Trend Component")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    axes[2].plot(daily_ts.index, seasonal, color='#2ca02c', lw=2)
    axes[2].set_title("3. Weekly Seasonality S(t)")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    axes[3].scatter(daily_ts.index, decomp.resid, color='#d62728', s=25)
    axes[3].axhline(0, color='gray', linestyle='--')
    axes[3].set_title("4. Stochastic Residual Component")
    axes[3].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "1_ts_decomposition.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=0.7, vmax=1.0)
    fig.colorbar(cax)
    cols = corr_matrix.columns
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=15)
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.3f}", ha='center', va='center', color='black', fontweight='bold')
    plt.title("Cross-Correlation Matrix (Revenue, Quantity, Shipping Fee)", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "2_correlation_matrix.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {0: 'tab:blue', 1: 'tab:red'}
    for cat in category_pivot.columns:
        cl = clusters[cat]
        ax.plot(category_pivot.index, category_pivot[cat], label=f"{cat} (Cl. {cl})",
                color=colors[cl], alpha=0.8, lw=1.8, linestyle='-' if cl == 0 else '--')
    ax.set_title("K-Means Dynamic Demand Clustering by Product Categories")
    ax.set_ylabel("Revenue ($)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "3_categories_cluster.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily_ts.index, daily_ts['Revenue'], label="Real Daily Revenue", color='black', lw=2)
    ax.plot(daily_ts.index, synthetic_series.values, label="Synthetic Generated Process",
            color='tab:green', lw=2, linestyle='--')
    ax.set_title(f"Adequacy Verification: Real vs Synthetic Time Series (KS-test p-value = {adequacy['ks_pvalue']:.4f})")
    ax.set_ylabel("Revenue ($)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "4_synthetic_validation.png"), dpi=300)
    plt.close()

    print("\n[6/6] Усі аналітичні графіки згенеровано та збережено у папку plots/")

if __name__ == "__main__":
    main()