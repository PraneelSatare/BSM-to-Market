"""
Fetch real-world market data for SPX (S&P 500 Index) European option chains
and underlying historical prices to benchmark Black-Scholes and PDE solvers.

Note: Unlike SPY (American style), SPX options are cash-settled and strictly
European style (exercised only at expiration), matching the Black-Scholes assumptions.
"""

import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_spx_data():
    print("=" * 72)
    print("Fetching Real European Market Data: ^SPX (S&P 500 Index)")
    print("=" * 72)

    ticker = yf.Ticker("^SPX")

    # 1. Historical underlying index prices for realized volatility
    print("  [1/4] Downloading 1-year historical daily prices for ^SPX …")
    hist = ticker.history(period="1y")
    hist_file = os.path.join(DATA_DIR, "spx_underlying_history.csv")
    hist.to_csv(hist_file)
    spot = float(hist["Close"].iloc[-1])
    print(f"        Latest SPX Spot Close: ${spot:.2f}")

    # Calculate 30-day and 1-year annualized realized volatility
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    realized_vol_1y = float(np.std(log_ret) * np.sqrt(252))
    realized_vol_30d = float(np.std(log_ret.tail(30)) * np.sqrt(252))
    print(f"        1-Year Realized Volatility: {realized_vol_1y * 100:.2f}%")
    print(f"        30-Day Realized Volatility: {realized_vol_30d * 100:.2f}%")

    # 2. Risk-free rate (3-month Treasury ^IRX) & S&P 500 dividend yield
    print("  [2/4] Fetching risk-free rate & dividend yield …")
    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        r_rate = float(irx["Close"].iloc[-1]) / 100.0
        print(f"        Current 13-week Treasury rate (^IRX): {r_rate * 100:.2f}%")
    except Exception:
        r_rate = 0.0409
        print(f"        Using default risk-free rate: {r_rate * 100:.2f}%")

    # S&P 500 continuous dividend yield (standard index yield ~1.25%)
    div_yield = 0.0125
    print(f"        S&P 500 Continuous Dividend yield (D0): {div_yield * 100:.2f}%")

    # 3. Available European option expirations
    expirations = ticker.options
    print(f"  [3/4] Found {len(expirations)} total European expiration dates.")

    today = datetime.date.today()
    target_days = [30, 60, 90]
    selected_expiries = []
    for td in target_days:
        best_exp = min(expirations, key=lambda d: abs((datetime.datetime.strptime(d, "%Y-%m-%d").date() - today).days - td))
        if best_exp not in selected_expiries:
            selected_expiries.append(best_exp)

    print(f"        Selected representative monthly maturities: {selected_expiries}")

    # 4. Fetch European option chains
    print("  [4/4] Downloading European option chains …")
    all_options = []
    for exp_str in selected_expiries:
        exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        T = dte / 365.25

        opt_chain = ticker.option_chain(exp_str)

        for opt_type, chain in [("call", opt_chain.calls), ("put", opt_chain.puts)]:
            df_opt = chain.copy()
            df_opt["optionType"] = opt_type
            df_opt["expiration"] = exp_str
            df_opt["dte"] = dte
            df_opt["T"] = T

            # Market price: use mid if bid/ask valid, else lastPrice
            has_bid_ask = (df_opt["bid"] > 0) & (df_opt["ask"] > df_opt["bid"])
            df_opt["marketPrice"] = np.where(has_bid_ask, 0.5 * (df_opt["bid"] + df_opt["ask"]), df_opt["lastPrice"])
            df_opt["hasBidAsk"] = has_bid_ask

            all_options.append(df_opt)

    df_all = pd.concat(all_options, ignore_index=True)

    # Filter liquid European options around spot (0.75 * S to 1.25 * S)
    df_filtered = df_all[
        (df_all["strike"] >= 0.75 * spot) &
        (df_all["strike"] <= 1.25 * spot) &
        (df_all["marketPrice"] > 0.50) &
        ((df_all["openInterest"].fillna(0) > 0) | (df_all["volume"].fillna(0) > 0) | df_all["hasBidAsk"])
    ].copy()

    df_filtered["spot"] = spot
    df_filtered["r"] = r_rate
    df_filtered["D0"] = div_yield
    df_filtered["realized_vol_1y"] = realized_vol_1y
    df_filtered["realized_vol_30d"] = realized_vol_30d
    df_filtered["moneyness"] = df_filtered["strike"] / spot

    out_file = os.path.join(DATA_DIR, "spx_options_market_data.csv")
    df_filtered.to_csv(out_file, index=False)
    print(f"\n[✓] Saved {len(df_filtered)} European option quotes to: {out_file}")
    print(f"    Underlying spot: ${spot:.2f}, Strikes: [{df_filtered['strike'].min():.0f}, {df_filtered['strike'].max():.0f}]")
    print(f"    Calls: {(df_filtered['optionType'] == 'call').sum()}, Puts: {(df_filtered['optionType'] == 'put').sum()}")
    print(f"    Expirations: {list(df_filtered['expiration'].unique())}")
    return out_file


if __name__ == "__main__":
    fetch_spx_data()
