from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(series, significance=0.05):
    result = adfuller(series.dropna())
    return {
        "test": "ADF",
        "statistic": result[0],
        "p_value": result[1],
        "stationary": result[1] <= significance
    }

def kpss_test(series, significance=0.05):
    statistic, p_value, _, _ = kpss(series.dropna(), regression='c')
    return {
        "test": "KPSS",
        "statistic": statistic,
        "p_value": p_value,
        "stationary": p_value > significance
    }

