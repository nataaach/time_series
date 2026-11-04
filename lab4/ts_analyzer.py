import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class TSAnalyzer:
    def __init__(self, series: pd.Series):
        self.series = series.asfreq('D', fill_value=0.0)

    def calculate_properties(self) -> dict:
        data = self.series.values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            adf_result = adfuller(data, autolag='AIC')
        
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'skewness': float(pd.Series(data).skew()),
            'kurtosis': float(pd.Series(data).kurtosis()),
            'adf_stat': float(adf_result[0]),
            'adf_pvalue': float(adf_result[1]),
            'is_stationary': bool(adf_result[1] < 0.05)
        }

    def decompose(self, period: int = 7):
        eff_period = min(period, max(2, len(self.series) // 3))
        decomposition = seasonal_decompose(
            self.series, 
            model='additive', 
            period=eff_period, 
            extrapolate_trend='period'
        )
        return decomposition

    @staticmethod
    def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        return df.corr(method='pearson')

    @staticmethod
    def cluster_categories(pivot_df: pd.DataFrame, n_clusters: int = 2) -> dict:
        clean_df = pivot_df.fillna(0.0)
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(clean_df.T)
        
        k = min(n_clusters, normalized_data.shape[0])
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_data)
        
        return {col: int(lbl) for col, lbl in zip(clean_df.columns, labels)}