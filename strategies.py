import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

## 0. Getting Stock history

# df = yf.Ticker("AAPL").history(period="1y")
# df.to_csv("AAPL 1 Year History", index=True)
# print(df)

# df = yf.Ticker("AAPL").history(period="max")
# df.to_csv("AAPL History", index=True)

## 1. Strategy Idea

def movingAverageCrossover():
    fast_ma = df["Close"].rolling(10).mean()
    slow_ma = df["Close"].rolling(30).mean()
    
    # Signal/position vector. The position at each bar
    df["Signal"] = np.where(fast_ma > slow_ma, 1, 0)

    df["Return"] = np.log(df["Close"]).diff().shift(-1)
    df["Strategy_Return"] = df["Signal"] * df["Return"]

    r = df["Strategy_Return"]
    profit_factor = r[r > 0].sum() / r[r < 0].abs().sum()
    sharpe_ratio = r.mean() / r.std()

    # print(fast_ma)
    # print(df)
    # print(profit_factor)
    # print(sharpe_ratio)

    fig, ax = plt.subplots()
    ax.plot(df["Close"], label="Close Price")
    ax.plot(fast_ma, label="10-Day Moving Avg.")
    ax.plot(slow_ma, label="30-Day Moving Avg.")
    
    fig.legend(loc='outside upper right')

    ax.set_ylabel("Price ($)")
    ax.set_xlabel("Date")

    plt.show()

# movingAverageCrossover()

def donchianBreakout(df, lookback):
    upper = df["Close"].rolling(lookback - 1).max().shift(1)
    lower = df["Close"].rolling(lookback - 1).min().shift(1)

    signal = pd.Series(np.full(len(df), np.nan), index=df.index)
    # print(signal)
    signal.loc[df["Close"] > upper] = 1
    signal.loc[df["Close"] < lower] = -1
    signal = signal.ffill()
    return signal

# print(donchianBreakout(4))

## 2. Optimizer

def optimizeDonchian(df):
    best_pf = 0
    best_lookback = -1

    r = np.log(df["Close"]).diff().shift(-1)
    for lookback in range(12, 169):
        signal = donchianBreakout(df, lookback)
        sig_rets = signal * r
        sig_pf = sig_rets[sig_rets > 0].sum() / sig_rets[sig_rets < 0].abs().sum()

        if sig_pf > best_pf:
            best_pf = lookback
            best_lookback = lookback

    return best_lookback, best_pf

# print(optimizeDonchian())

if __name__ == "__main__":
    df = pd.read_csv("AAPL History", index_col="Date")
    df.index = df.index.astype("datetime64[s, UTC-05:00]")

    ## 3. Development Data
    df = df[ (df.index.year > 2020) & (df.index.year < 2026) ]
    # print(df.index.year)

    best_lookback, best_pf = optimizeDonchian(df)
    
    print(best_lookback, best_pf)
    # best_lookback = 12, best_pf = 12

    signal = donchianBreakout(df, best_lookback)

    df["r"] = np.log(df["Close"]).diff().shift(-1)
    df["donch_r"] = df["r"] * signal

    df["donch_r"].cumsum().plot()
    plt.show()