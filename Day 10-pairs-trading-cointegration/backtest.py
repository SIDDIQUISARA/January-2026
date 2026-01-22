from data_loader import load_price_data
from cointegration_test import cointegration_test
from pairs_trading_strategy import calculate_spread, generate_signals
import pandas as pd

def run_backtest(ticker1, ticker2):
    data = load_price_data(ticker1, ticker2)

    pvalue = cointegration_test(data[ticker1], data[ticker2])
    print(f"Cointegration p-value: {pvalue}")

    if pvalue > 0.05:
        print("Pairs are NOT cointegrated")
        return

    spread, hedge_ratio = calculate_spread(data[ticker1], data[ticker2])
    signals = generate_signals(spread)

    returns = spread.diff()
    signals["pnl"] = signals["position"] * returns
    signals["cumulative_pnl"] = signals["pnl"].cumsum()

    print("Pairs are cointegrated")
    print(signals.tail())

    return spread, signals

if __name__ == "__main__":
    run_backtest("AAPL", "MSFT")
