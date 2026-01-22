import numpy as np
import pandas as pd
import statsmodels.api as sm

def calculate_spread(series1, series2):
    series2_const = sm.add_constant(series2)
    model = sm.OLS(series1, series2_const).fit()
    hedge_ratio = model.params[1]
    spread = series1 - hedge_ratio * series2
    return spread, hedge_ratio

def generate_signals(spread, window=20):
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    z_score = (spread - mean) / std

    signals = pd.DataFrame(index=spread.index)
    signals["z_score"] = z_score
    signals["position"] = 0
    signals.loc[z_score < -1, "position"] = 1    # Long spread
    signals.loc[z_score > 1, "position"] = -1    # Short spread
    signals.loc[abs(z_score) < 0.5, "position"] = 0

    signals["position"] = signals["position"].shift(1)
    return signals
