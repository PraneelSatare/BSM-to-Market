"""
Research Paper Extensions to the Black-Scholes PDE:
Progressive Error Reduction Analysis on European SPX Options.

Evaluates 5 models progressively relaxing assumptions:
  0. Baseline: Classical Black-Scholes with Historical Realized Volatility
  1. Leland (1985): Nonlinear PDE with Transaction Costs & Discrete Hedging (c = 20 bps, Δt = 1/252)
  2. ATM Calibrated: Forward Market Implied Volatility Term Structure (ATM Variance Risk Premium)
  3. Smile-Interpolated BS: Per-expiry polynomial smile interpolation in log-moneyness ln(K/F)
  4. Merton (1976): Jump-Diffusion PIDE capturing market crash probability and fat tails

Both in-sample calibration and 5-fold cross-validation out-of-sample metrics are reported.
"""

import sys
import os
import math
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

from bs_analytical import bs_call, bs_put
from iv_utils import compute_iv, bs_price, pcp_implied_forward


def merton_price(S, E, T, r, q, is_call, sig_m, lam, mu_j, delta_j, N_terms=15):
    """Merton (1976) Jump-Diffusion analytical mixture series."""
    k_j = np.exp(mu_j + 0.5 * delta_j**2) - 1.0
    lam_p = lam * (1.0 + k_j)
    total = np.zeros_like(E, dtype=float)
    for n in range(N_terms):
        prob = np.exp(-lam_p * T) * ((lam_p * T)**n) / float(math.factorial(n))
        sig_n = np.sqrt(sig_m**2 + n * delta_j**2 / T)
        r_n = r - lam * k_j + n * np.log(1.0 + k_j) / T
        total += prob * bs_price(S, E, T, r_n, sig_n, q, is_call)
    return total


def run_research_models():
    data_file = os.path.join(DATA_DIR, "spx_options_market_data.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing {data_file}. Run fetch_market_data.py first.")

    raw_df = pd.read_csv(data_file)
    raw_df = raw_df[(raw_df["bid"] > 0) & (raw_df["ask"] > raw_df["bid"])].copy().reset_index(drop=True)

    spot = float(raw_df["spot"].iloc[0])
    r = float(raw_df["r"].iloc[0])
    sig_real = float(raw_df["realized_vol_1y"].iloc[0])

    # Infer put-call parity forward and dividend yield per expiry
    expirations = sorted(raw_df["expiration"].unique())
    exp_fwd = {}
    exp_q = {}
    for exp in expirations:
        sub_e = raw_df[raw_df["expiration"] == exp]
        T_e = float(sub_e["T"].iloc[0])
        c_sub = sub_e[sub_e["optionType"] == "call"]
        p_sub = sub_e[sub_e["optionType"] == "put"]
        F_imp, q_imp = pcp_implied_forward(c_sub, p_sub, r, T_e, spot)
        exp_fwd[exp] = F_imp
        exp_q[exp] = q_imp

    raw_df["F_pcp"] = raw_df["expiration"].map(exp_fwd)
    raw_df["q_pcp"] = raw_df["expiration"].map(exp_q)
    raw_df["moneyness_fwd"] = raw_df["strike"] / raw_df["F_pcp"]
    raw_df["log_moneyness"] = np.log(raw_df["strike"] / raw_df["F_pcp"])

    # Standard quantitative finance practice: OTM options (Puts for K <= F, Calls for K > F)
    raw_df["is_otm"] = (
        ((raw_df["optionType"] == "put") & (raw_df["strike"] <= raw_df["F_pcp"])) |
        ((raw_df["optionType"] == "call") & (raw_df["strike"] > raw_df["F_pcp"]))
    )

    # Focus empirical model benchmarking on the clean OTM contracts
    df = raw_df[raw_df["is_otm"]].copy().reset_index(drop=True)

    # Extract implied volatility for all OTM options
    ivs = []
    for _, row in df.iterrows():
        is_call_i = row["optionType"] == "call"
        iv = compute_iv(spot, row["strike"], row["T"], r, row["q_pcp"], row["marketPrice"], is_call_i)
        ivs.append(iv)
    df["impliedVol"] = ivs

    is_call = (df["optionType"] == "call").values
    market_price = df["marketPrice"].values
    spreads = df["ask"] - df["bid"]
    median_spread = float(spreads.median())
    mean_spread = float(spreads.mean())

    print("=" * 78)
    print("RESEARCH EXTENSIONS: Relaxing Black-Scholes Assumptions on European ^SPX")
    print("=" * 78)
    print(f"  Underlying Asset: ^SPX Index (S = ${spot:.2f}, r = {r*100:.2f}%)")
    print(f"  Historical Realized Volatility: σ = {sig_real*100:.2f}%")
    print(f"  Benchmark Dataset: {len(df)} Liquid OTM Quotes across {len(expirations)} Maturities")
    print(f"  OTM Bid-Ask Spread: Median = ${median_spread:.2f}, Mean = ${mean_spread:.2f}")
    print("=" * 78)

    predictions = {}

    # -------------------------------------------------------------------------
    # Model 0: Baseline Black-Scholes (Constant Historical Realized Volatility)
    # -------------------------------------------------------------------------
    print(f"  [1/5] Model 0: Baseline Black-Scholes (1-Yr Realized σ = {sig_real*100:.2f}%) …")
    p_base = bs_price(df["spot"].values, df["strike"].values, df["T"].values,
                      r, sig_real, df["q_pcp"].values, is_call)
    predictions["0. Baseline (Realized σ)"] = p_base

    # -------------------------------------------------------------------------
    # Model 1: Leland (1985) Transaction Costs & Discrete Hedging Friction
    # Relaxes: Frictionless continuous delta-hedging assumption
    # -------------------------------------------------------------------------
    c_cost = 0.0020
    dt_hedge = 1.0 / 252.0
    leland_boost = np.sqrt(2.0 / np.pi) * (c_cost / (sig_real * np.sqrt(dt_hedge)))
    sig_leland = sig_real * np.sqrt(1.0 + leland_boost)
    print(f"  [2/5] Model 1: Leland (1985) Transaction Costs (c = 20 bps, daily) …")
    print(f"        Leland Effective Volatility: {sig_leland * 100:.2f}% (boosted from {sig_real*100:.2f}%)")
    p_leland = bs_price(df["spot"].values, df["strike"].values, df["T"].values,
                        r, sig_leland, df["q_pcp"].values, is_call)
    predictions["1. Leland (1985) Friction"] = p_leland

    # -------------------------------------------------------------------------
    # Model 2: ATM Implied Volatility Calibration
    # Relaxes: Backward-looking historical volatility as a forward predictor
    # -------------------------------------------------------------------------
    print("  [3/5] Model 2: ATM Market Volatility Term Structure (Variance Risk Premium) …")
    atm_vols = {}
    for exp in expirations:
        sub_atm = df[(df["expiration"] == exp) & (np.abs(df["moneyness_fwd"] - 1.0) <= 0.02)]
        atm_vols[exp] = float(sub_atm["impliedVol"].mean())
        print(f"        Maturity {exp}: ATM Market IV = {atm_vols[exp] * 100:.2f}%")

    p_atm = np.zeros(len(df))
    for exp in expirations:
        mask = (df["expiration"] == exp).values
        sub = df[mask]
        p_atm[mask] = bs_price(sub["spot"].values, sub["strike"].values, sub["T"].values,
                               r, atm_vols[exp], sub["q_pcp"].values, (sub["optionType"] == "call").values)
    predictions["2. ATM Calibrated"] = p_atm

    # -------------------------------------------------------------------------
    # Model 3: Smile-Interpolated Black-Scholes (Per-Expiry Polynomial Smile)
    # Relaxes: Constant volatility across strikes (captures empirical skew and smile)
    # -------------------------------------------------------------------------
    print("  [4/5] Model 3: Smile-Interpolated Black-Scholes (Quadratic in ln(K/F)) …")
    p_smile = np.zeros(len(df))
    for exp in expirations:
        mask = (df["expiration"] == exp).values
        sub = df[mask]
        log_m = sub["log_moneyness"].values
        poly = np.polyfit(log_m, sub["impliedVol"].values, deg=2)
        vol_fit = np.polyval(poly, log_m)
        p_smile[mask] = bs_price(sub["spot"].values, sub["strike"].values, sub["T"].values,
                                 r, vol_fit, sub["q_pcp"].values, (sub["optionType"] == "call").values)
    predictions["3. Smile-Interpolated BS"] = p_smile

    # -------------------------------------------------------------------------
    # Model 4: Merton (1976) Jump-Diffusion Model
    # Relaxes: Continuous Gaussian paths (Poisson jump process with Gaussian jump sizes)
    # -------------------------------------------------------------------------
    print("  [5/5] Model 4: Merton (1976) Jump-Diffusion PIDE (Crash Jumps) …")
    def merton_obj(params):
        sig_c, lam_j, mu_j, del_j = params
        pred = merton_price(df["spot"].values, df["strike"].values, df["T"].values,
                            r, df["q_pcp"].values, is_call, sig_c, lam_j, mu_j, del_j)
        return np.mean((market_price - pred)**2)

    res_m = minimize(merton_obj, [0.10, 0.50, -0.15, 0.12],
                     bounds=[(0.04, 0.22), (0.1, 2.5), (-0.30, 0.0), (0.05, 0.40)],
                     method="L-BFGS-B")
    sig_c, lam_j, mu_j, del_j = res_m.x
    print(f"        Calibrated: λ = {lam_j:.2f}/yr, Mean Jump = {mu_j*100:.1f}%, Jump σ = {del_j*100:.1f}%, Cont. σ = {sig_c*100:.2f}%")
    p_merton = merton_price(df["spot"].values, df["strike"].values, df["T"].values,
                            r, df["q_pcp"].values, is_call, sig_c, lam_j, mu_j, del_j)
    predictions["4. Merton (1976) Jumps"] = p_merton

    # -------------------------------------------------------------------------
    # Out-of-Sample 5-Fold Cross-Validation for Calibrated Models
    # -------------------------------------------------------------------------
    print("\n  Running 5-Fold Cross-Validation for Calibrated Models (Out-of-Sample) …")
    np.random.seed(42)
    n = len(df)
    indices = np.random.permutation(n)
    folds = np.array_split(indices, 5)

    oos_preds_smile = np.zeros(n)
    oos_preds_merton = np.zeros(n)

    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Smile CV
        for exp in expirations:
            tr_sub = train_df[train_df["expiration"] == exp]
            te_sub = test_df[test_df["expiration"] == exp]
            if len(te_sub) == 0:
                continue
            poly = np.polyfit(tr_sub["log_moneyness"].values, tr_sub["impliedVol"].values, deg=2)
            vol_te = np.polyval(poly, te_sub["log_moneyness"].values)
            oos_preds_smile[te_sub.index] = bs_price(
                te_sub["spot"].values, te_sub["strike"].values, te_sub["T"].values,
                r, vol_te, te_sub["q_pcp"].values, (te_sub["optionType"] == "call").values
            )

        # Merton CV
        is_call_tr = (train_df["optionType"] == "call").values
        def m_obj_cv(params):
            sc, lj, mj, dj = params
            pr = merton_price(train_df["spot"].values, train_df["strike"].values, train_df["T"].values,
                              r, train_df["q_pcp"].values, is_call_tr, sc, lj, mj, dj)
            return np.mean((train_df["marketPrice"].values - pr)**2)

        res_cv = minimize(m_obj_cv, [0.10, 0.50, -0.15, 0.12],
                          bounds=[(0.04, 0.22), (0.1, 2.5), (-0.30, 0.0), (0.05, 0.40)],
                          method="L-BFGS-B")
        is_call_te = (test_df["optionType"] == "call").values
        oos_preds_merton[test_df.index] = merton_price(
            test_df["spot"].values, test_df["strike"].values, test_df["T"].values,
            r, test_df["q_pcp"].values, is_call_te, *res_cv.x
        )

    # -------------------------------------------------------------------------
    # Performance Evaluation & Error Reduction Reporting
    # -------------------------------------------------------------------------
    print("\n" + "=" * 82)
    print("IN-SAMPLE MODEL EVALUATION ON OTM EUROPEAN SPX OPTIONS")
    print("=" * 82)
    print(f"{'Model / Extension':<30} {'RMSE ($)':>10} {'MAE ($)':>10} {'Max Err ($)':>12} {'RMSE Drop (%)':>15}")
    print("-" * 82)

    metrics = []
    rmse_base = np.sqrt(np.mean((market_price - predictions["0. Baseline (Realized σ)"])**2))

    for name, pred in predictions.items():
        err = market_price - pred
        rmse = np.sqrt(np.mean(err**2))
        mae = np.mean(np.abs(err))
        max_err = np.max(np.abs(err))
        drop_pct = (1.0 - (rmse / rmse_base)) * 100.0
        metrics.append({
            "model": name,
            "rmse": rmse,
            "mae": mae,
            "max_err": max_err,
            "drop_pct": drop_pct
        })
        drop_str = "0.00% (Baseline)" if "Baseline" in name else f"-{drop_pct:5.2f}%"
        print(f"{name:<30} {rmse:10.2f} {mae:10.2f} {max_err:12.2f} {drop_str:>15}")

    print("=" * 82)

    # Out-of-sample summary
    rmse_smile_oos = np.sqrt(np.mean((market_price - oos_preds_smile)**2))
    mae_smile_oos = np.mean(np.abs(market_price - oos_preds_smile))
    rmse_merton_oos = np.sqrt(np.mean((market_price - oos_preds_merton)**2))
    mae_merton_oos = np.mean(np.abs(market_price - oos_preds_merton))

    print("\n" + "=" * 82)
    print("OUT-OF-SAMPLE 5-FOLD CROSS-VALIDATION (Calibrated Models)")
    print("=" * 82)
    print(f"  Smile-Interpolated BS : In-Sample RMSE = ${metrics[3]['rmse']:.2f} | Out-of-Sample RMSE = ${rmse_smile_oos:.2f} (MAE: ${mae_smile_oos:.2f})")
    print(f"  Merton (1976) Jumps   : In-Sample RMSE = ${metrics[4]['rmse']:.2f} | Out-of-Sample RMSE = ${rmse_merton_oos:.2f} (MAE: ${mae_merton_oos:.2f})")
    print("=" * 82)

    # Moneyness breakdown for each model
    print("\n" + "=" * 82)
    print("RMSE by Forward Moneyness Regime across Models ($)")
    print("-" * 82)
    print(f"{'Model':<30} {'Deep OTM (K/F < 0.9)':>22} {'ATM (|K/F-1| ≤ 2%)':>22} {'OTM Calls (K/F > 1.05)':>24}")
    print("-" * 82)

    mask_deep_puts = (df["optionType"] == "put") & (df["moneyness_fwd"] < 0.90)
    mask_atm = np.abs(df["moneyness_fwd"] - 1.0) <= 0.02
    mask_otm_calls = (df["optionType"] == "call") & (df["moneyness_fwd"] > 1.05)

    for name, pred in predictions.items():
        err = market_price - pred
        rmse_dp = np.sqrt(np.mean(err[mask_deep_puts]**2)) if mask_deep_puts.sum() > 0 else 0.0
        rmse_atm = np.sqrt(np.mean(err[mask_atm]**2)) if mask_atm.sum() > 0 else 0.0
        rmse_oc = np.sqrt(np.mean(err[mask_otm_calls]**2)) if mask_otm_calls.sum() > 0 else 0.0
        print(f"{name:<30} {rmse_dp:22.2f} {rmse_atm:22.2f} {rmse_oc:24.2f}")

    print("=" * 82)

    # -------------------------------------------------------------------------
    # Plot 1: Bar Chart of Progressive Error Reduction
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    names_short = ["Baseline\n(Realized σ)", "Leland\n(Friction)", "ATM Calib.\n(VRP)", "Smile-Interp.\n(BS)", "Merton\n(Jumps)"]
    rmses = [m["rmse"] for m in metrics]
    maes = [m["mae"] for m in metrics]
    colors = ["#c0392b", "#e67e22", "#f39c12", "#27ae60", "#2980b9"]

    bars1 = ax1.bar(names_short, rmses, color=colors, edgecolor="black", width=0.55)
    ax1.set_ylabel("Root Mean Square Error ($)", fontsize=11)
    ax1.set_title("Progressive RMSE Reduction on Traded OTM ^SPX Options", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars1, metrics):
        h = bar.get_height()
        label = f"${h:.2f}" if "Baseline" in m["model"] else f"${h:.2f}\n(-{m['drop_pct']:.1f}%)"
        ax1.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    bars2 = ax2.bar(names_short, maes, color=colors, edgecolor="black", width=0.55)
    ax2.axhline(median_spread, color="black", linestyle="--", lw=1.2, label=f"Median Bid-Ask Spread (${median_spread:.2f})")
    ax2.set_ylabel("Mean Absolute Error ($)", fontsize=11)
    ax2.set_title("Mean Absolute Error (MAE) vs. Bid-Ask Spread", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"${h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.suptitle("Impact of Relaxing Black-Scholes PDE Assumptions with Literature Extensions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, "research_models_error_reduction.png"), dpi=150)
    plt.close(fig)
    print("[✓] Saved research_models_error_reduction.png")

    # -------------------------------------------------------------------------
    # Plot 2: Model Price Fit Comparison Across Strikes
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sub_exp = expirations[0]
    sub_df = df[df["expiration"] == sub_exp].copy().sort_values("strike")
    idx_sub = sub_df.index
    F_sub = exp_fwd[sub_exp]

    sub_calls = sub_df[sub_df["optionType"] == "call"]
    sub_puts = sub_df[sub_df["optionType"] == "put"]

    # Panel 1: OTM Call Prices (K > F)
    if len(sub_calls) > 0:
        ax1.plot(sub_calls["strike"], sub_calls["marketPrice"], "ko", ms=4, label="Market Traded Price", alpha=0.85)
        ax1.plot(sub_calls["strike"], predictions["0. Baseline (Realized σ)"][sub_calls.index], "r--", lw=1.3, label="Baseline (Realized σ)")
        ax1.plot(sub_calls["strike"], predictions["2. ATM Calibrated"][sub_calls.index], "y-.", lw=1.3, label="ATM Calibrated")
        ax1.plot(sub_calls["strike"], predictions["3. Smile-Interpolated BS"][sub_calls.index], "g-", lw=1.8, label="Smile-Interp. BS")
        ax1.plot(sub_calls["strike"], predictions["4. Merton (1976) Jumps"][sub_calls.index], "b:", lw=1.8, label="Merton (1976)")
        ax1.axvline(F_sub, color="grey", linestyle=":", alpha=0.7, label=f"Forward F=${F_sub:.0f}")
        ax1.set_xlabel("Strike E ($)")
        ax1.set_ylabel("OTM Call Option Price ($)")
        ax1.set_title(f"OTM Call Option Fit across Strikes ({sub_exp})")
        ax1.legend(fontsize=8.5)
        ax1.grid(True, alpha=0.3)

    # Panel 2: OTM Put Pricing Error across Strikes (K <= F)
    idx_p = sub_puts.index
    ax2.plot(sub_puts["strike"], market_price[idx_p] - predictions["0. Baseline (Realized σ)"][idx_p], "r--o", ms=3.5, lw=1.2, label="Baseline (Realized σ)")
    ax2.plot(sub_puts["strike"], market_price[idx_p] - predictions["1. Leland (1985) Friction"][idx_p], "m-.^", ms=3.5, lw=1.2, label="Leland (1985)")
    ax2.plot(sub_puts["strike"], market_price[idx_p] - predictions["2. ATM Calibrated"][idx_p], "y-s", ms=3.5, lw=1.2, label="ATM Calibrated")
    ax2.plot(sub_puts["strike"], market_price[idx_p] - predictions["3. Smile-Interpolated BS"][idx_p], "g-x", ms=4, lw=1.8, label="Smile-Interp. BS")
    ax2.plot(sub_puts["strike"], market_price[idx_p] - predictions["4. Merton (1976) Jumps"][idx_p], "b:d", ms=3.5, lw=1.5, label="Merton (1976)")
    ax2.axhline(0.0, color="black", lw=1.0)
    ax2.axvline(F_sub, color="grey", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Strike E ($)")
    ax2.set_ylabel("Pricing Error: V_market − V_model ($)")
    ax2.set_title(f"OTM Put Option Pricing Error across Strikes ({sub_exp})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Progressive Error Elimination on European SPX Options", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, "research_models_price_fit.png"), dpi=150)
    plt.close(fig)
    print("[✓] Saved research_models_price_fit.png")

    print("\nDone.")


if __name__ == "__main__":
    run_research_models()
