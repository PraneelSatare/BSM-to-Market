"""
Black-Scholes formulae for European options.
Zhu et al., Derivative Securities and Difference Methods (2nd ed.), Sect. 2.6.5.

T denotes time to expiry (T - t in the book).
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, E, T, r, sigma, D0=0.0):
    S = np.asarray(S, dtype=float)
    d1 = (np.log(S / E) + (r - D0 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_call(S, E, T, r, sigma, D0=0.0):
    """c = S e^{-D0 T} N(d1) - E e^{-rT} N(d2)   — Eq. (2.90)."""
    S = np.asarray(S, dtype=float)
    d1, d2 = _d1_d2(S, E, T, r, sigma, D0)
    return S * np.exp(-D0 * T) * norm.cdf(d1) - E * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, E, T, r, sigma, D0=0.0):
    """p = E e^{-rT} N(-d2) - S e^{-D0 T} N(-d1)   — Eq. (2.91)."""
    S = np.asarray(S, dtype=float)
    d1, d2 = _d1_d2(S, E, T, r, sigma, D0)
    return E * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-D0 * T) * norm.cdf(-d1)


def bs_call_delta(S, E, T, r, sigma, D0=0.0):
    """Δ = e^{-D0 T} N(d1)."""
    d1, _ = _d1_d2(S, E, T, r, sigma, D0)
    return np.exp(-D0 * T) * norm.cdf(d1)


def bs_call_gamma(S, E, T, r, sigma, D0=0.0):
    """Γ = e^{-D0 T} n(d1) / (S σ √T)."""
    S = np.asarray(S, dtype=float)
    d1, _ = _d1_d2(S, E, T, r, sigma, D0)
    return np.exp(-D0 * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
