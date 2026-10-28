# Black-Scholes PDE: From Numerical Heat Solvers to Real-Market SPX Options

A quantitative finance framework implementing the Black-Scholes PDE transformation into the 1D Heat Equation (Zhu et al.), numerical solving via Finite Difference Methods (Explicit, Crank-Nicolson, Fully Implicit), empirical European market validation on 1,277 live S&P 500 (`^SPX`) options, and progressive error relaxation using literature models (Leland, Carr & Wu, Merton, Dupire).

---

## Project Structure

```
BSM-to-Market/
├── src/                          # Core numerical algorithms and market calibration models
│   ├── bs_analytical.py          # Analytical Black-Scholes European pricing and Greeks
│   ├── heat_transform.py         # Coordinate mappings between financial and heat domains
│   ├── fdm_heat.py               # Finite Difference Solvers (Explicit, Crank-Nicolson, Implicit)
│   ├── main.py                   # PDE validation, stability, and convergence studies
│   ├── fetch_market_data.py      # Live European ^SPX option chain & historical price downloader
│   ├── market_validation.py      # Empirical European market benchmarking & IV solver
│   └── model_extensions.py       # Literature models (Leland 1985, Carr-Wu 2009, Merton 1976, Dupire 1994)
│
├── data/                         # Real-world market datasets
│   ├── spx_options_market_data.csv # 1,277 traded European SPX option quotes across 3 maturities
│   └── spx_underlying_history.csv  # 1-year historical daily closes for S&P 500 index
│
├── figures/                      # High-resolution generated plots and diagnostic figures
│   ├── prices_comparison.png     # FDM curves overlaid on analytical Black-Scholes
│   ├── error_analysis.png        # Time and space convergence rates (orders O(Δτ²) and O(Δx²))
│   ├── heat_solution_surface.png # 3D spacetime heat solution surface u(x, τ̄)
│   ├── stability_demo.png        # Von Neumann numerical stability criterion (α ≤ 1/2)
│   ├── constant_alpha_analysis.png # Simultaneous (M, N) refinement at constant α = 0.40
│   ├── market_volatility_smile.png # Empirical European SPX volatility skew vs. flat realized σ
│   ├── market_pricing_errors.png # European market vs. constant-σ pricing errors across moneyness
│   ├── market_iv_surface.png     # 3D Implied Volatility Surface σ_imp(K/S, T) for ^SPX
│   ├── research_models_error_reduction.png # Progressive RMSE/MAE drops across literature models
│   └── research_models_price_fit.png       # Fit comparison across strikes & elimination of put skew error
│
├── docs/                         # Theoretical reference literature
│   └── DERIVATIVE_SECURITIES_AND_DIFFERENCE_METHODS_SECOND_--_You-lan_Z.pdf
│
├── .gitignore                    # Python and virtual environment exclusions
├── requirements.txt              # Dependencies (numpy, scipy, matplotlib, pandas, yfinance)
└── README.md
```

---

## 1. Problem Formulation

The European option price $V(S, t)$ satisfies the Black-Scholes PDE (Zhu et al. Eq. 2.73):

$$
V_t + \tfrac12\sigma^2 S^2 V_{SS} + (r - D_0) S V_S - rV = 0, \qquad V(S,T) = \max(S - E, 0)
$$

---

## 2. Heat Transformation (Zhu et al. §2.6.1)

Applying the coordinate change:

$$
x = \ln(S/E) + \big(r - D_0 - \tfrac12\sigma^2\big)(T-t), \qquad
\bar\tau = \tfrac12\sigma^2 (T-t), \qquad
V = E\, e^{-r(T-t)}\, u(x,\bar\tau),
$$

transforms the PDE into the constant-coefficient heat equation:

$$
u_{\bar\tau} = u_{xx}, \qquad u(x,0) = \max(e^x - 1, 0)
$$

with boundary conditions on $[x_l, x_u]$ ($k = 2(r-D_0)/\sigma^2$):

$$
u(x_l,\bar\tau) = 0, \qquad u(x_u,\bar\tau) = \big(e^{x_u - (k-1)\bar\tau} - 1\big) e^{k\bar\tau}
$$

---

## 3. Finite Difference Schemes (§7.1)

Discretized on mesh $x_m = x_l + m\Delta x$, $\bar\tau_n = n\Delta\bar\tau$, mesh ratio $\alpha = \Delta\bar\tau/\Delta x^2$:

* **Explicit (FTCS):** $u_m^{n+1} = \alpha u_{m+1}^n + (1 - 2\alpha)u_m^n + \alpha u_{m-1}^n$. Conditionally stable iff $\alpha \le \tfrac12$.
* **Crank-Nicolson (Trapezoidal):** $A\mathbf u^{n+1} = B\mathbf u^n + \mathbf b^n$. Tridiagonal system solved via Thomas algorithm in $\mathcal{O}(M)$ time. Unconditionally stable with $\mathcal{O}(\Delta\bar\tau^2 + \Delta x^2)$ accuracy.
* **Fully Implicit (BTCS):** Unconditionally stable with $\mathcal{O}(\Delta\bar\tau + \Delta x^2)$ accuracy.

---

## 4. Execution & Quick Start

```bash
# 1. Activate virtual environment
source .ugp/bin/activate

# 2. Run numerical PDE validation and convergence studies
python src/main.py

# 3. Fetch live European SPX option chains
python src/fetch_market_data.py

# 4. Run empirical market validation
python src/market_validation.py

# 5. Run literature extensions (Leland, Carr-Wu, Merton, Dupire)
python src/model_extensions.py
```

---

## 5. Numerical Verification Results

Tested on normalized parameters ($E = 100, T = 1.0, r = 0.10, D_0 = 0.05, \sigma = 0.20$):

```
Scheme                     V(E,0)   Max |err|        RMSE  ATM rel (%)
----------------------------------------------------------------------
Explicit (7.8)           9.940409   1.104e-03   6.540e-04      0.00496
Crank-Nicolson (7.9)     9.938706   2.220e-03   9.684e-04      0.02210
Fully implicit*          9.937001   3.944e-03   1.634e-03      0.03925
Analytical (2.90)        9.940903

observed order in time : implicit ≈ 1.05,  Crank-Nicolson ≈ 2.00
observed order in space: Crank-Nicolson ≈ 2.02, 2.00, 2.00, 2.00
observed order at constant α = 0.40: Explicit ≈ 2.00,  Crank-Nicolson ≈ 2.00,  Fully implicit ≈ 2.00
```

---

## 6. Real Market Validation (^SPX European Options)

Tested against **1,277 live European option quotes** on the S&P 500 Index ($S = \$7,744.06, r = 4.09\%, D_0 = 1.25\%$):

* **FDM vs. Black-Scholes Consistency:** FDM reproduces Black-Scholes analytical prices within $\$1.52$ ($< 0.1\%$) on actual market scales, confirming numerical correctness.
* **Variance Risk Premium:** Average ATM implied volatility is **$14.48\%$**, trading at an excess **$+1.81\%$** over historical realized volatility ($12.68\%$).
* **Volatility Skew:** OTM crash-protection puts trade at implied volatilities up to **$42\%+$**, producing large pricing errors under classical constant-$\sigma$ Black-Scholes (Baseline RMSE: **$\$35.14$**).

---

## 7. Literature Extensions & Progressive Error Reduction

Systematically relaxing Black-Scholes assumptions using seminal quantitative finance papers yields the following error reductions on the 1,277 European SPX options:

| Model / Literature Extension | Assumption Relaxed | Overall RMSE ($) | Overall MAE ($) | RMSE Reduction |
| :--- | :--- | :---: | :---: | :---: |
| **0. Baseline Black-Scholes** | Classical model with Realized $\sigma = 12.68\%$ | **$35.14** | **$27.92** | *0.0% (Baseline)* |
| **1. Leland (1985)** | Frictionless continuous hedging ($c = 20\text{ bps}$, daily) | **$32.09** | **$26.28** | **-8.68%** |
| **2. ATM Calibrated (Carr-Wu 2009)** | Backward-looking historical $\sigma \to$ forward ATM term structure | **$31.82** | **$25.97** | **-9.43%** |
| **3. Merton (1976) Jump-Diffusion** | Continuous Gaussian paths $\to$ Poisson crash jumps ($\lambda = 0.58$, $\mu_J = -14.1\%$) | **$16.31** | **$10.82** | **-53.58%** |
| **4. Dupire (1994) Local Volatility** | Constant $\sigma$ across strikes $\to$ local smile surface $\sigma(K, T)$ | **$17.84** | **$11.10** | **-49.23%** |

---

## References

1. Y.-l. Zhu, X. Wu, I-L. Chern, Z.-z. Sun, *Derivative Securities and Difference Methods*, 2nd ed., Springer, 2013.
2. F. Black, M. Scholes, "The Pricing of Options and Corporate Liabilities", *Journal of Political Economy*, 81(3), 637–654, 1973.
3. H. E. Leland, "Option Pricing and Replication with Transactions Costs", *The Journal of Finance*, 40(5), 1283–1301, 1985.
4. R. C. Merton, "Option Pricing When Underlying Stock Returns are Discontinuous", *Journal of Financial Economics*, 3(1–2), 125–144, 1976.
5. B. Dupire, "Pricing with a Smile", *Risk*, 7(1), 18–20, 1994.
6. P. Carr, L. Wu, "The Variance Risk Premium", *The Review of Financial Studies*, 22(3), 1311–1341, 2009.
