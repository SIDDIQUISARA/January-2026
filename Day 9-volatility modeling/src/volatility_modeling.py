"""
Day 9 - Volatility Modeling
GARCH(1,1) Implementation

Author: Your Name
Description:
    This script demonstrates how to model and forecast financial time series
    volatility using a GARCH(1,1) model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf


def load_data(ticker="AAPL", start="2018-01-01", end="2024-01-01"):
    """
    Download adjusted close prices and compute log returns.
    """
    data = yf.download(ticker, start=start, end=end)
    prices = data["Adj Close"]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def fit_garch_model(returns):
    """
    Fit a GARCH(1,1) model to returns.
    """
    model = arch_model(
        returns * 100,  # scale for numerical stability
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    model_fit = model.fit(disp="off")
    return model_fit


def plot_conditional_volatility(model_fit, returns):
    """
    Plot conditional volatility from GARCH model.
    """
    conditional_vol = model_fit.conditional_volatility

    plt.figure(figsize=(12, 6))
    plt.plot(conditional_vol, label="Conditional Volatility", color="red")
    plt.title("GARCH(1,1) Conditional Volatility")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()


def forecast_volatility(model_fit, horizon=10):
    """
    Forecast future volatility.
    """
    forecast = model_fit.forecast(horizon=horizon)
    variance_forecast = forecast.variance.iloc[-1]
    volatility_forecast = np.sqrt(variance_forecast)

    return volatility_forecast


def main():
    returns = load_data("AAPL")
    model_fit = fit_garch_model(returns)

    print(model_fit.summary())

    plot_conditional_volatility(model_fit, returns)

    vol_forecast = forecast_volatility(model_fit, horizon=5)
    print("\n5-Day Volatility Forecast:")
    print(vol_forecast)


if __name__ == "__main__":
    main()
