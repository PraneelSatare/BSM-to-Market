"""
Fetch real-world market data for SPX (S&P 500 Index) European option chains
and underlying historical prices to benchmark Black-Scholes and PDE solvers.

Note:
  - SPX options are cash-settled and strictly European style (exercised only
    at expiration), matching Black-Scholes boundary conditions.
  - CAUTION: Running this script queries live Yahoo Finance data and overwrites
    data/spx_options_market_data.csv and data/spx_underlying_history.csv,
    which will change all empirical numbers reported in the README.
"""

import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

from iv_utils import pcp_implied_forward


def fetch_spx_data(reference_date=None):
    """
    Fetch S&P 500 index history and representative monthly European option chains.
    Optionally pass a reference_date (datetime.date) to calculate time to maturity T;
    defaults to datetime.date.today().
    """
    today = reference_date or datetime.date.today()
    snapshot_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print("=" * 72)
    print("Fetching Real European Market Data: ^SPX (S&P 500 Index)")
    print(f"  Reference Date for DTE calculation: {today}")
    print("=" * 72)

    ticker = yf.Ticker("^SPX")

    # 1. Historical underlying index prices for realized volatility
    print("  [1/4] Downloading 1-year historical daily prices for ^SPX …")
    hist = ticker.history(period="1y")
    hist_file = DATA_DIR / "spx_underlying_history.csv"
    hist.to_csv(hist_file)
    spot = float(hist["Close"].iloc[-1])
    print(f"        Latest SPX Spot Close: ${spot:.2f}")

    # Calculate 30-day and 1-year annualized realized volatility
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    realized_vol_1y = float(np.std(log_ret) * np.sqrt(252))
    realized_vol_30d = float(np.std(log_ret.tail(30)) * np.sqrt(252))
    print(f"        1-Year Realized Volatility: {realized_vol_1y * 100:.2f}%")
    print(f"        30-Day Realized Volatility: {realized_vol_30d * 100:.2f}%")

    # 2. Risk-free rate (13-week Treasury Bill ^IRX)
    print("  [2/4] Fetching risk-free rate from ^IRX (13-week Treasury) …")
    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        r_rate = float(irx["Close"].iloc[-1]) / 100.0
        print(f"        Current 13-week Treasury rate (^IRX): {r_rate * 100:.2f}%")
    except Exception:
        r_rate = 0.04088
        print(f"        Using default risk-free rate: {r_rate * 100:.2f}%")

    # 3. Available European option expirations
    expirations = ticker.options
    print(f"  [3/4] Found {len(expirations)} total European expiration dates.")

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

            # Require active quotes with bid > 0 and ask > bid
            has_bid_ask = (df_opt["bid"] > 0) & (df_opt["ask"] > df_opt["bid"])
            df_opt["hasBidAsk"] = has_bid_ask
            df_opt["bidAskSpread"] = np.where(has_bid_ask, df_opt["ask"] - df_opt["bid"], np.nan)
            df_opt["marketPrice"] = np.where(has_bid_ask, 0.5 * (df_opt["bid"] + df_opt["ask"]), np.nan)

            all_options.append(df_opt)

    df_all = pd.concat(all_options, ignore_index=True)

    # Filter liquid European options with active bid/ask around spot (0.75 * S to 1.25 * S)
    df_filtered = df_all[
        (df_all["hasBidAsk"]) &
        (df_all["strike"] >= 0.75 * spot) &
        (df_all["strike"] <= 1.25 * spot) &
        (df_all["marketPrice"] > 0.50)
    ].copy().reset_index(drop=True)

    df_filtered["spot"] = spot
    df_filtered["r"] = r_rate
    df_filtered["realized_vol_1y"] = realized_vol_1y
    df_filtered["realized_vol_30d"] = realized_vol_30d
    df_filtered["snapshot_timestamp"] = snapshot_timestamp

    # Infer risk-neutral forward price and dividend yield per expiry from put-call parity
    fwd_map = {}
    q_map = {}
    for exp_str in selected_expiries:
        sub_exp = df_filtered[df_filtered["expiration"] == exp_str]
        calls_sub = sub_exp[sub_exp["optionType"] == "call"]
        puts_sub = sub_exp[sub_exp["optionType"] == "put"]
        T_sub = float(sub_exp["T"].iloc[0]) if len(sub_exp) > 0 else 1.0
        F_imp, q_imp = pcp_implied_forward(calls_sub, puts_sub, r_rate, T_sub, spot)
        fwd_map[exp_str] = F_imp
        q_map[exp_str] = q_imp
        print(f"        Expiry {exp_str} (T = {T_sub:.4f}y): PCP Forward F = ${F_imp:.2f}, Implied Dividend Yield q = {q_imp * 100:.2f}%")

    df_filtered["F_pcp"] = df_filtered["expiration"].map(fwd_map)
    df_filtered["q_pcp"] = df_filtered["expiration"].map(q_map)
    df_filtered["D0"] = df_filtered["q_pcp"]
    df_filtered["moneyness_spot"] = df_filtered["strike"] / spot
    df_filtered["moneyness_fwd"] = df_filtered["strike"] / df_filtered["F_pcp"]
    df_filtered["log_moneyness"] = np.log(df_filtered["strike"] / df_filtered["F_pcp"])

    # Out-of-the-money classification relative to forward price
    df_filtered["is_otm"] = (
        ((df_filtered["optionType"] == "put") & (df_filtered["strike"] <= df_filtered["F_pcp"])) |
        ((df_filtered["optionType"] == "call") & (df_filtered["strike"] > df_filtered["F_pcp"]))
    )

    out_file = DATA_DIR / "spx_options_market_data.csv"
    df_filtered.to_csv(out_file, index=False)
    print(f"\n[✓] Saved {len(df_filtered)} clean European option quotes to: {out_file}")
    print(f"    Underlying spot: ${spot:.2f}, Strikes: [{df_filtered['strike'].min():.0f}, {df_filtered['strike'].max():.0f}]")
    print(f"    Calls: {(df_filtered['optionType'] == 'call').sum()}, Puts: {(df_filtered['optionType'] == 'put').sum()}")
    print(f"    OTM Quotes (for smile fitting): {df_filtered['is_otm'].sum()}")
    print(f"    Expirations: {list(df_filtered['expiration'].unique())}")
    return out_file


if __name__ == "__main__":
    fetch_spx_data()
