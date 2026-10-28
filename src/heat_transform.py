"""
Black-Scholes -> heat equation transformation, Eq. (2.78) of Zhu et al. (Sect. 2.6.1):

    x = ln(S/E) + (r - D0 - σ²/2)(T - t),   τ̄ = σ²(T - t)/2,   V = E e^{-r(T-t)} u(x, τ̄)

giving  u_τ̄ = u_xx,  u(x, 0) = max(e^x - 1, 0)   — Eq. (2.77)/(8.5).
"""

import numpy as np


class BSHeatTransform:

    def __init__(self, E, T, r, sigma, D0=0.0):
        self.E, self.T, self.r, self.sigma, self.D0 = E, T, r, sigma, D0
        self.k = 2.0 * (r - D0) / sigma**2
        self.mu = r - D0 - 0.5 * sigma**2

    def tau_bar_max(self):
        """τ̄ at t = 0."""
        return 0.5 * self.sigma**2 * self.T

    def t_to_tau_bar(self, t):
        return 0.5 * self.sigma**2 * (self.T - np.asarray(t, dtype=float))

    def tau_bar_to_t(self, tau_bar):
        return self.T - 2.0 * np.asarray(tau_bar, dtype=float) / self.sigma**2

    def S_to_x(self, S, t=0.0):
        return np.log(np.asarray(S, dtype=float) / self.E) + self.mu * (self.T - t)

    def x_to_S(self, x, t=0.0):
        return self.E * np.exp(np.asarray(x, dtype=float) - self.mu * (self.T - t))

    def u_to_V(self, u, t=0.0):
        return self.E * np.exp(-self.r * (self.T - t)) * np.asarray(u, dtype=float)

    def V_to_u(self, V, t=0.0):
        return np.exp(self.r * (self.T - t)) * np.asarray(V, dtype=float) / self.E

    def call_ic(self, x):
        return np.maximum(np.exp(np.asarray(x, dtype=float)) - 1.0, 0.0)

    def call_bc_lower(self, tau_bar):
        """u(x_l, τ̄) = 0   — Sect. 8.2.1."""
        return 0.0

    def call_bc_upper(self, x_u, tau_bar):
        """u(x_u, τ̄) = (e^{x_u - (k-1)τ̄} - 1) e^{kτ̄},  k = 2(r - D0)/σ²   — Sect. 8.2.1."""
        k = self.k
        return (np.exp(x_u - (k - 1.0) * tau_bar) - 1.0) * np.exp(k * tau_bar)
