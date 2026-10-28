"""
Finite-difference solvers for u_τ = u_xx on [x_l, x_u] × [0, tau_max]  (Zhu et al., Sect. 7.1).

Grid: x_m = x_l + mΔx, τ_n = nΔτ, mesh ratio α = Δτ/Δx².
  * Explicit, Eq. (7.8)        — stable iff α ≤ 1/2 (Sect. 7.2.1)
  * Crank-Nicolson, Eq. (7.9)  — A u^{n+1} = B u^n + b^n, Eq. (7.10)
  * Fully implicit             — not in the book; for comparison only
"""

import warnings
import numpy as np
from scipy.linalg import solve_banded


class HeatFDMSolver:

    def __init__(self, x_l, x_u, tau_max, M, N, f, f_l, f_u):
        self.x_l, self.x_u = x_l, x_u
        self.tau_max = tau_max
        self.M, self.N = M, N

        self.dx = (x_u - x_l) / M
        self.dtau = tau_max / N
        self.alpha = self.dtau / self.dx**2

        self.x = x_l + self.dx * np.arange(M + 1)
        self.tau = self.dtau * np.arange(N + 1)

        self.f, self.f_l, self.f_u = f, f_l, f_u

    def _init_grid(self):
        u = np.zeros((self.M + 1, self.N + 1))
        u[:, 0] = self.f(self.x)
        return u

    def solve_explicit(self):
        a, M = self.alpha, self.M
        if a > 0.5:
            warnings.warn(f"Explicit scheme: α = {a:.4f} > 1/2 — unstable.",
                          RuntimeWarning, stacklevel=2)
        u = self._init_grid()
        for n in range(self.N):
            u[1:M, n + 1] = a * u[2:M + 1, n] + (1.0 - 2.0 * a) * u[1:M, n] + a * u[0:M - 1, n]
            u[0, n + 1] = self.f_l(self.tau[n + 1])
            u[M, n + 1] = self.f_u(self.tau[n + 1])
        return u

    def solve_crank_nicolson(self):
        a, M = self.alpha, self.M
        u = self._init_grid()

        # A = tridiag(-α/2, 1+α, -α/2) in banded form
        A = np.empty((3, M - 1))
        A[0, :] = -a / 2.0
        A[1, :] = 1.0 + a
        A[2, :] = -a / 2.0

        for n in range(self.N):
            u[0, n + 1] = self.f_l(self.tau[n + 1])
            u[M, n + 1] = self.f_u(self.tau[n + 1])
            # B u^n (includes old-time boundary terms of b^n)
            rhs = (a / 2.0) * u[0:M - 1, n] + (1.0 - a) * u[1:M, n] + (a / 2.0) * u[2:M + 1, n]
            # new-time boundary terms of b^n
            rhs[0] += (a / 2.0) * u[0, n + 1]
            rhs[-1] += (a / 2.0) * u[M, n + 1]
            u[1:M, n + 1] = solve_banded((1, 1), A, rhs)
        return u

    def solve_implicit(self):
        a, M = self.alpha, self.M
        u = self._init_grid()

        A = np.empty((3, M - 1))
        A[0, :] = -a
        A[1, :] = 1.0 + 2.0 * a
        A[2, :] = -a

        for n in range(self.N):
            u[0, n + 1] = self.f_l(self.tau[n + 1])
            u[M, n + 1] = self.f_u(self.tau[n + 1])
            rhs = u[1:M, n].copy()
            rhs[0] += a * u[0, n + 1]
            rhs[-1] += a * u[M, n + 1]
            u[1:M, n + 1] = solve_banded((1, 1), A, rhs)
        return u
