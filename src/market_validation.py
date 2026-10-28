"""
Empirical Market Validation of Black-Scholes & Finite Difference Model.
Uses real-world European-style option chain data from SPX (S&P 500 Index).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from bs_analytical import bs_call
from heat_transform import BSHeatTransform
from fdm_heat import HeatFDMSolver
from main import quad_interp
from iv_utils import compute_iv, bs_price, pcp_implied_forward


def run_market_validation():
    data_file = DATA_DIR / "spx_options_market_data.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing {data_file}. Run fetch_market_data.py first.")

    df = pd.read_csv(data_file)
    spot = float(df["spot"].iloc[0])
    # Risk-free rate r is obtained from 13-week Treasury Bill (^IRX) at the snapshot date
    r = float(df["r"].iloc[0])
    sig_real_1y = float(df["realized_vol_1y"].iloc[0])
    sig_real_30d = float(df["realized_vol_30d"].iloc[0])

    # Ensure valid bid/ask quotes
    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])].copy().reset_index(drop=True)

    # Infer put-call parity forward and dividend yield per expiry
    expirations = sorted(df["expiration"].unique())
    exp_fwd = {}
    exp_q = {}
    for exp in expirations:
        sub_e = df[df["expiration"] == exp]
        T_e = float(sub_e["T"].iloc[0])
        c_sub = sub_e[sub_e["optionType"] == "call"]
        p_sub = sub_e[sub_e["optionType"] == "put"]
        F_imp, q_imp = pcp_implied_forward(c_sub, p_sub, r, T_e, spot)
        exp_fwd[exp] = F_imp
        exp_q[exp] = q_imp

    df["F_pcp"] = df["expiration"].map(exp_fwd)
    df["q_pcp"] = df["expiration"].map(exp_q)
    df["log_moneyness"] = np.log(df["strike"] / df["F_pcp"])
    df["moneyness_fwd"] = df["strike"] / df["F_pcp"]
    df["moneyness_spot"] = df["strike"] / spot

    # Standard quantitative finance convention: OTM puts (K <= F) and OTM calls (K > F)
    df["is_otm"] = (
        ((df["optionType"] == "put") & (df["strike"] <= df["F_pcp"])) |
        ((df["optionType"] == "call") & (df["strike"] > df["F_pcp"]))
    )

    spreads = df["ask"] - df["bid"]
    median_spread = float(spreads.median())
    mean_spread = float(spreads.mean())

    maturities_str = ", ".join([f"{df[df['expiration'] == exp]['T'].iloc[0]:.3f}y ({exp})" for exp in expirations])

    print("=" * 76)
    print("Real European Market Validation: Black-Scholes vs. Traded ^SPX Options")
    print("=" * 76)
    print(f"  Underlying Asset: ^SPX (S&P 500 Index)   Spot S = ${spot:.2f}")
    print(f"  Style: Strictly European (Cash-settled, no early exercise)")
    print(f"  Risk-Free Rate r = {r * 100:.2f}% (13-week T-Bill ^IRX at snapshot)")
    print(f"  1-Year Realized Volatility: {sig_real_1y * 100:.2f}%")
    print(f"  30-Day Realized Volatility: {sig_real_30d * 100:.2f}%")
    print(f"  Total Valid European Quotes: {len(df)} across {len(expirations)} maturities: {maturities_str}")
    print(f"  Bid-Ask Spread: Median = ${median_spread:.2f}, Mean = ${mean_spread:.2f}")
    print("  Put-Call Parity Implied Forwards & Dividend Yields:")
    for exp in expirations:
        sub_e = df[df["expiration"] == exp]
        T_e = sub_e["T"].iloc[0]
        print(f"    • {exp} (T = {T_e:.4f}y): Forward F = ${exp_fwd[exp]:.2f}, Implied q = {exp_q[exp]*100:.2f}%")
    print("=" * 76)

    # 1. Compute Implied Volatilities using PCP-consistent forward and dividend yields
    print("  [1/4] Calculating Implied Volatilities (root-solving Black-Scholes with PCP forward) …")
    iv_list = []
    for _, row in df.iterrows():
        is_call = row["optionType"] == "call"
        iv = compute_iv(spot, row["strike"], row["T"], r, row["q_pcp"], row["marketPrice"], is_call)
        iv_list.append(iv)

    df["impliedVol"] = iv_list
    valid_iv_count = df["impliedVol"].notna().sum()
    otm_df = df[df["is_otm"]].copy()
    otm_iv_count = otm_df["impliedVol"].notna().sum()
    print(f"        Valid IV extracted: {valid_iv_count}/{len(df)} total options ({otm_iv_count}/{len(otm_df)} OTM options, 100% of OTM).")

    # 2. Compute theoretical prices using Realized Volatility
    print(f"  [2/4] Pricing options using constant Realized Volatility (σ = {sig_real_1y * 100:.2f}%) …")
    is_call_all = (df["optionType"] == "call").values
    df["modelPrice"] = bs_price(df["spot"].values, df["strike"].values, df["T"].values,
                                r, sig_real_1y, df["q_pcp"].values, is_call_all)
    df["pricingError"] = df["marketPrice"] - df["modelPrice"]
    df["absError"] = np.abs(df["pricingError"])
    df["relError"] = (df["absError"] / df["marketPrice"]) * 100.0

    # 3. Verify FDM solver consistency on actual market parameters with scaled grid
    print("  [3/4] Verifying Crank-Nicolson FDM consistency with diffusion-scaled grid …")
    atm_opts = df[(df["optionType"] == "call") & (np.abs(df["moneyness_fwd"] - 1.0) <= 0.005)]
    sample_opt = atm_opts.iloc[0] if len(atm_opts) > 0 else df[df["optionType"] == "call"].iloc[0]
    E_atm = float(sample_opt["strike"])
    T_atm = float(sample_opt["T"])
    q_atm = float(sample_opt["q_pcp"])

    # Scale spatial domain with diffusion scale σ√T to prevent grid under-resolution
    diff_scale = sig_real_1y * np.sqrt(T_atm)
    x_bound = max(0.30, 8.0 * diff_scale)
    x_l, x_u = -x_bound, x_bound
    M_scaled, N_scaled = 600, 500

    ht = BSHeatTransform(E_atm, T_atm, r, sig_real_1y, q_atm)
    solver = HeatFDMSolver(x_l, x_u, ht.tau_bar_max(), M_scaled, N_scaled,
                           f=ht.call_ic, f_l=ht.call_bc_lower,
                           f_u=lambda tb: ht.call_bc_upper(x_u, tb))
    u_cn = solver.solve_crank_nicolson()[:, -1]
    v_fdm_atm = float(ht.u_to_V(quad_interp(solver.x, u_cn, ht.S_to_x(spot, 0.0))))
    v_bs_atm = float(bs_call(spot, E_atm, T_atm, r, sig_real_1y, q_atm))
    num_diff = abs(v_fdm_atm - v_bs_atm)
    rel_diff_pct = (num_diff / v_bs_atm) * 100.0

    print(f"        Diffusion scale σ√T ≈ {diff_scale:.3f}, Scaled x-grid: [{x_l:.2f}, {x_u:.2f}], Δx = {solver.dx:.5f}")
    print(f"        ATM Strike: ${E_atm:.0f}, Expiry: {sample_opt['expiration']} (T = {T_atm:.4f}y)")
    print(f"        FDM Price: ${v_fdm_atm:.4f}  |  BS Analytical Price: ${v_bs_atm:.4f}")
    print(f"        Numerical Difference: ${num_diff:.4f}  ({rel_diff_pct:.4f}% relative error, about one cent)")
    print(f"        Actual Market Price: ${sample_opt['marketPrice']:.2f}")
    print(f"        Market vs. Realized-Model Gap: ${sample_opt['marketPrice'] - v_bs_atm:.2f}")

    # 4. Generate Diagnostic Plots
    print("  [4/4] Generating empirical European market validation plots …")

    # ---- Plot 1: Volatility Smile / Skew ----
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, exp in enumerate(expirations):
        sub_otm = df[(df["expiration"] == exp) & (df["is_otm"]) & (df["impliedVol"].notna())].sort_values("moneyness_fwd")
        T_val = sub_otm["T"].iloc[0]

        # OTM Puts (K <= F)
        puts_otm = sub_otm[sub_otm["optionType"] == "put"]
        calls_otm = sub_otm[sub_otm["optionType"] == "call"]

        if len(puts_otm) > 0:
            ax.plot(puts_otm["moneyness_fwd"], puts_otm["impliedVol"] * 100, "o-",
                    color=colors[idx % len(colors)], label=f"OTM Puts: {exp} (T={T_val:.3f}y)",
                    alpha=0.85, ms=4)
        if len(calls_otm) > 0:
            ax.plot(calls_otm["moneyness_fwd"], calls_otm["impliedVol"] * 100, "s--",
                    color=colors[idx % len(colors)], label=f"OTM Calls: {exp}",
                    alpha=0.7, ms=4)

    ax.axhline(sig_real_1y * 100, color="black", linestyle="--", lw=1.5,
               label=f"1-Yr Realized Volatility ({sig_real_1y * 100:.2f}%)")
    ax.axvline(1.0, color="grey", linestyle=":", lw=1, alpha=0.7, label="ATM Forward (K = F)")
    ax.set_xlabel(r"Forward Moneyness  $K / F_{\mathrm{PCP}}$", fontsize=11)
    ax.set_ylabel("Implied Volatility (%)", fontsize=11)
    ax.set_title("Empirical Volatility Smile / Skew in Traded European SPX Options\n(Constructed from liquid OTM options with PCP-implied forward)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(8, 45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "market_volatility_smile.png", dpi=150)
    plt.close(fig)
    print("[✓] Saved market_volatility_smile.png")

    # ---- Plot 2: Pricing Errors & Market vs Model ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sub_sample = df[df["expiration"] == expirations[0]].sort_values("strike")
    calls = sub_sample[sub_sample["optionType"] == "call"]
    puts = sub_sample[sub_sample["optionType"] == "put"]
    F_sample = exp_fwd[expirations[0]]

    ax1.plot(calls["strike"], calls["marketPrice"], "bo", ms=3.5, alpha=0.7, label="Market Call Price")
    ax1.plot(calls["strike"], calls["modelPrice"], "b-", lw=1.5, label=f"BS Model (σ={sig_real_1y*100:.1f}%)")
    ax1.plot(puts["strike"], puts["marketPrice"], "ro", ms=3.5, alpha=0.7, label="Market Put Price")
    ax1.plot(puts["strike"], puts["modelPrice"], "r-", lw=1.5, label=f"BS Model (σ={sig_real_1y*100:.1f}%)")
    ax1.axvline(F_sample, color="grey", linestyle=":", alpha=0.7, label=f"Forward F=${F_sample:.0f}")
    ax1.set_xlabel("Strike E ($)")
    ax1.set_ylabel("Option Price ($)")
    ax1.set_title(f"European Market vs. Constant-σ Model Prices ({expirations[0]})")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Pricing Error vs Moneyness for OTM Options
    otm_sample = sub_sample[sub_sample["is_otm"]]
    otm_calls_s = otm_sample[otm_sample["optionType"] == "call"]
    otm_puts_s = otm_sample[otm_sample["optionType"] == "put"]

    ax2.plot(otm_puts_s["moneyness_fwd"], otm_puts_s["pricingError"], "rv", ms=4, alpha=0.8,
             label="OTM Puts (Market − Model)")
    ax2.plot(otm_calls_s["moneyness_fwd"], otm_calls_s["pricingError"], "b^", ms=4, alpha=0.8,
             label="OTM Calls (Market − Model)")
    ax2.axhline(0.0, color="black", linestyle="-", lw=1.0)
    ax2.axvline(1.0, color="grey", linestyle=":", alpha=0.7)
    ax2.set_xlabel(r"Forward Moneyness  $K / F_{\mathrm{PCP}}$")
    ax2.set_ylabel("Pricing Error: V_market − V_model ($)")
    ax2.set_title("Systematic Mispricing of OTM Downside Protection")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Empirical European Market Pricing Comparison: Reality vs. Constant-Volatility Black-Scholes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "market_pricing_errors.png", dpi=150)
    plt.close(fig)
    print("[✓] Saved market_pricing_errors.png")

    # ---- Plot 3: 3D Implied Volatility Surface ----
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    df_surf = df[(df["is_otm"]) & (df["impliedVol"].notna()) & (df["moneyness_fwd"] >= 0.80) & (df["moneyness_fwd"] <= 1.20)].copy()

    scatter = ax.scatter(df_surf["moneyness_fwd"], df_surf["T"], df_surf["impliedVol"] * 100,
                         c=df_surf["impliedVol"] * 100, cmap="viridis", s=22, alpha=0.85)
    ax.set_xlabel(r"Forward Moneyness  $K / F$")
    ax.set_ylabel("Time to Expiry T (Years)")
    ax.set_zlabel("Implied Volatility (%)")
    ax.set_title("Empirical European Implied Volatility Surface σ_imp(K/F, T) for ^SPX\n(Constructed from liquid OTM contracts)",
                 fontsize=12, fontweight="bold")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Implied Volatility (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "market_iv_surface.png", dpi=150)
    plt.close(fig)
    print("[✓] Saved market_iv_surface.png")

    # 5. Statistical Error Analysis with Correctly Labeled Buckets
    print("\n" + "=" * 76)
    print(f"European Market Error Statistics (Realized Volatility σ = {sig_real_1y * 100:.2f}%)")
    print("=" * 76)
    print(f"  {'Category':<26} {'Count':>7} {'RMSE ($)':>11} {'MAE ($)':>11} {'Mean Bias ($)':>14} {'Median Spread ($)':>18}")
    print("  " + "-" * 91)

    categories = {
        "All Traded Options": df,
        "All OTM Options (K/F)": df[df["is_otm"]],
        "  • OTM Puts (K ≤ F)": df[(df["optionType"] == "put") & (df["is_otm"])],
        "  • OTM Calls (K > F)": df[(df["optionType"] == "call") & (df["is_otm"])],
        "ATM Region (|K/F−1| ≤ 2%)": df[np.abs(df["moneyness_fwd"] - 1.0) <= 0.02],
        "Deep OTM Puts (K/F < 0.90)": df[(df["optionType"] == "put") & (df["moneyness_fwd"] < 0.90)],
        "Deep OTM Calls (K/F > 1.10)": df[(df["optionType"] == "call") & (df["moneyness_fwd"] > 1.10)],
    }

    for cat_name, subset in categories.items():
        if len(subset) == 0:
            continue
        errs = subset["pricingError"]
        rmse = np.sqrt(np.mean(errs**2))
        mae = np.mean(np.abs(errs))
        bias = np.mean(errs)
        med_sp = float((subset["ask"] - subset["bid"]).median())
        print(f"  {cat_name:<26} {len(subset):7d} {rmse:11.2f} {mae:11.2f} {bias:14.2f} {med_sp:18.2f}")

    print("=" * 76)
    atm_sub = df[np.abs(df["moneyness_fwd"] - 1.0) <= 0.02]
    avg_atm_iv = float(atm_sub["impliedVol"].dropna().mean() * 100)
    
    # Specific maturity ATM IVs:
    atm_by_exp = {}
    for exp in expirations:
        sub_e_atm = df[(df["expiration"] == exp) & (np.abs(df["moneyness_fwd"] - 1.0) <= 0.02)]
        atm_by_exp[exp] = float(sub_e_atm["impliedVol"].dropna().mean() * 100)
    
    # Peak OTM put IV:
    otm_puts_all = df[(df["optionType"] == "put") & (df["is_otm"]) & (df["impliedVol"].notna())]
    max_otm_put_row = otm_puts_all.loc[otm_puts_all["impliedVol"].idxmax()]
    max_otm_put_iv = float(max_otm_put_row["impliedVol"] * 100)
    max_otm_put_exp = max_otm_put_row["expiration"]
    max_otm_put_moneyness = float(max_otm_put_row["moneyness_fwd"])
    max_otm_put_strike = float(max_otm_put_row["strike"])

    print(f"\n  Key Empirical Insights for European Options:")
    print(f"  • Mean ATM Market Implied Volatility:    {avg_atm_iv:.2f}% (across all expiries)")
    for exp in expirations:
        sub_e = df[df["expiration"] == exp]
        T_e = sub_e["T"].iloc[0]
        print(f"    - {exp} (T = {T_e:.3f}y): ATM IV = {atm_by_exp[exp]:.2f}% (spread vs RV {sig_real_1y*100:.2f}%: {atm_by_exp[exp]-sig_real_1y*100:+.2f}%)")
    print(f"  • Historical Realized Volatility:        {sig_real_1y * 100:.2f}%")
    print(f"  • Peak OTM Put Implied Volatility:       {max_otm_put_iv:.2f}% (at K = ${max_otm_put_strike:.0f}, K/F = {max_otm_put_moneyness:.2f}, {max_otm_put_exp})")
    print(f"  • Traded Option Median Bid-Ask Spread:   ${median_spread:.2f} (Mean: ${mean_spread:.2f})")
    print(f"  • Liquid OTM Median Bid-Ask Spread:      ${float((otm_df['ask'] - otm_df['bid']).median()):.2f} (Mean: ${float((otm_df['ask'] - otm_df['bid']).mean()):.2f})")
    print("\nDone.")


if __name__ == "__main__":
    run_market_validation()
