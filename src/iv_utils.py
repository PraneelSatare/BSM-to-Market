"""
Shared implied-volatility and forward-inference utilities.

Provides
--------
pcp_implied_forward : infer risk-neutral forward from put-call parity
compute_iv          : Brent's method Black-Scholes IV solver
bs_price            : vectorized BS pricing for mixed calls/puts
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bs_analytical import bs_call, bs_put


def pcp_implied_forward(calls_df, puts_df, r, T, spot, atm_band=0.03):
    """
    Infer the risk-neutral forward price F from put-call parity:

        C(K) - P(K) = [F - K] exp(-rT)
        =>  F = [C(K) - P(K)] exp(rT) + K

    Uses the median of ATM-region estimates for robustness.

    Parameters
    ----------
    calls_df, puts_df : DataFrame
        Must contain columns ``strike`` and ``marketPrice``.
    r, T, spot : float
        Risk-free rate, time to expiry, and current spot price.
    atm_band : float
        Fraction of spot that defines the ATM band (default 3 %).

    Returns
    -------
    F : float
        PCP-implied forward price.
    q : float
        Implied continuous dividend yield  q = r - ln(F/S) / T.
    """
    calls = calls_df[["strike", "marketPrice"]].drop_duplicates("strike").set_index("strike")
    puts = puts_df[["strike", "marketPrice"]].drop_duplicates("strike").set_index("strike")
    common = calls.index.intersection(puts.index)

    if len(common) == 0:
        F = spot * np.exp(r * T)
        return F, 0.0

    c = calls.loc[common, "marketPrice"].values.astype(float)
    p = puts.loc[common, "marketPrice"].values.astype(float)
    K = common.values.astype(float)

    F_per_strike = (c - p) * np.exp(r * T) + K

    atm = (K >= spot * (1.0 - atm_band)) & (K <= spot * (1.0 + atm_band))
    F = float(np.median(F_per_strike[atm])) if atm.sum() >= 3 else float(np.median(F_per_strike))
    q = r - np.log(F / spot) / T

    return F, q


def compute_iv(S, E, T, r, q, market_price, is_call, vol_lo=0.005, vol_hi=3.0):
    """
    Numerically invert the European Black-Scholes formula via Brent's method.

    Returns implied volatility (float), or ``NaN`` if the market price is at
    or below discounted intrinsic, or if root-finding fails.
    """
    S, E, T, r, q = float(S), float(E), float(T), float(r), float(q)
    market_price = float(market_price)

    if is_call:
        intrinsic = max(0.0, S * np.exp(-q * T) - E * np.exp(-r * T))
        fn = bs_call
    else:
        intrinsic = max(0.0, E * np.exp(-r * T) - S * np.exp(-q * T))
        fn = bs_put

    if market_price <= intrinsic + 1e-3:
        return np.nan

    try:
        return brentq(lambda v: fn(S, E, T, r, v, q) - market_price,
                      vol_lo, vol_hi, xtol=1e-6, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def bs_price(S, E, T, r, vol, q, is_call):
    """Vectorized Black-Scholes price for mixed calls and puts."""
    return np.where(is_call, bs_call(S, E, T, r, vol, q), bs_put(S, E, T, r, vol, q))
