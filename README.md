# Black-Scholes to Market: PDE Finite Difference Solvers, Empirical SPX Calibration, & Literature Error Relaxation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Quantitative Finance](https://img.shields.io/badge/Field-Quantitative%20Finance-emerald.svg)](#)
[![PDE Solvers](https://img.shields.io/badge/Numerics-FDM%20%7C%20Crank--Nicolson-orange.svg)](#)

A quantitative finance framework connecting mathematical PDE theory to real-world market reality. This repository implements:
1. The analytical **Black-Scholes-Merton (1973)** PDE and its transformation into the **1D Heat Diffusion Equation** (Zhu et al., 2013).
2. Numerical **Finite Difference Methods (FDM)** — Explicit FTCS, Crank-Nicolson, and Fully Implicit BTCS — with rigorous von Neumann stability and spacetime convergence proofs ($\mathcal{O}(\Delta x^2), \mathcal{O}(\Delta\tau^2)$).
3. **Empirical European Market Validation** on **1,277 live S&P 500 (`^SPX`)** index options, analyzing the empirical volatility smile and variance risk premium.
4. **Progressive Error Relaxation** using seminal quantitative finance papers (**Leland 1985**, **Carr & Wu 2009**, **Merton 1976**, **Dupire 1994**), cutting real-market pricing errors by up to **53.6%**.

---

## Table of Contents

- [Project Architecture](#project-architecture)
- [Theoretical Formulation](#theoretical-formulation)
  - [The Black-Scholes PDE](#the-black-scholes-pde)
  - [Transformation to Constant-Coefficient Heat Equation](#transformation-to-constant-coefficient-heat-equation)
- [Numerical Schemes & Convergence](#numerical-schemes--convergence)
  - [Finite Difference Implementations](#finite-difference-implementations)
  - [Numerical Stability & Spacetime Surface](#numerical-stability--spacetime-surface)
- [Empirical Market Validation (^SPX Index Options)](#empirical-market-validation-spx-index-options)
  - [Why European ^SPX Options?](#why-european-spx-options)
  - [Volatility Smile & Baseline Errors](#volatility-smile--baseline-errors)
- [Literature Extensions & Progressive Error Relaxation](#literature-extensions--progressive-error-relaxation)
  - [Relaxation Benchmark Results](#relaxation-benchmark-results)
  - [Model Progression Visualizations](#model-progression-visualizations)
  - [Economic Interpretation of Residual Error](#economic-interpretation-of-residual-error)
- [Installation & Quick Start](#installation--quick-start)
- [References](#references)
- [License](#license)

---

## Project Architecture

```
BSM-to-Market/
├── src/                          # Core numerical algorithms and calibration models
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
├── requirements.txt              # Production dependencies
└── README.md
```

---

## Theoretical Formulation

### The Black-Scholes PDE

Under risk-neutral valuation with continuous dividend yield $D_0$, the price of a European option $V(S, t)$ satisfies (Zhu et al. Eq. 2.73):

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r - D_0) S \frac{\partial V}{\partial S} - rV = 0, \qquad V(S,T) = \max(S - E, 0)$$

### Transformation to Constant-Coefficient Heat Equation

Following Zhu et al. (§2.6.1), we apply dimensionless coordinate transformations:

$$x = \ln(S/E) + \left(r - D_0 - \frac{1}{2}\sigma^2\right)(T-t), \qquad \bar\tau = \frac{1}{2}\sigma^2 (T-t), \qquad V(S,t) = E\, e^{-r(T-t)}\, u(x,\bar\tau)$$

This transforms the variable-coefficient parabolic PDE into the canonical 1D heat diffusion equation:

$$\frac{\partial u}{\partial \bar\tau} = \frac{\partial^2 u}{\partial x^2}, \qquad u(x,0) = \max\left(e^x - 1, 0\right)$$

with truncated Dirichlet boundaries on $[x_l, x_u]$ ($k = \frac{2(r-D_0)}{\sigma^2}$):

$$u(x_l,\bar\tau) = 0, \qquad u(x_u,\bar\tau) = \left(e^{x_u - (k-1)\bar\tau} - 1\right) e^{k\bar\tau}$$

---

## Numerical Schemes & Convergence

### Finite Difference Implementations

Discretizing on spatial mesh $x_m = x_l + m\Delta x$ and temporal mesh $\bar\tau_n = n\Delta\bar\tau$ with mesh ratio $\alpha = \frac{\Delta\bar\tau}{\Delta x^2}$:

1. **Explicit FTCS (Forward Time, Centered Space):**
   $$u_m^{n+1} = \alpha u_{m+1}^n + (1 - 2\alpha)u_m^n + \alpha u_{m-1}^n$$
   *Stability:* Conditionally stable if and only if $\alpha \le \frac{1}{2}$ (Von Neumann criterion).

2. **Crank-Nicolson (Trapezoidal Scheme):**
   $$-\frac{\alpha}{2}u_{m-1}^{n+1} + (1+\alpha)u_m^{n+1} - \frac{\alpha}{2}u_{m+1}^{n+1} = \frac{\alpha}{2}u_{m-1}^n + (1-\alpha)u_m^n + \frac{\alpha}{2}u_{m+1}^n$$
   *Properties:* Unconditionally stable, tridiagonal linear system solved in $\mathcal{O}(M)$ time via Thomas algorithm, second-order accurate $\mathcal{O}(\Delta\bar\tau^2 + \Delta x^2)$.

3. **Fully Implicit BTCS (Backward Time, Centered Space):**
   $$-\alpha u_{m-1}^{n+1} + (1+2\alpha)u_m^{n+1} - \alpha u_{m+1}^{n+1} = u_m^n$$
   *Properties:* Unconditionally stable, $\mathcal{O}(\Delta\bar\tau + \Delta x^2)$ accuracy.

| Scheme | $V(E,0)$ | Max Absolute Error | RMSE | ATM Relative Error (%) | Observed Time Order | Observed Space Order |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Explicit FTCS** | 9.940409 | $1.104 \times 10^{-3}$ | $6.540 \times 10^{-4}$ | **0.00496%** | — | $2.00$ |
| **Crank-Nicolson** | 9.938706 | $2.220 \times 10^{-3}$ | $9.684 \times 10^{-4}$ | **0.02210%** | **2.00** | **2.00** |
| **Fully Implicit** | 9.937001 | $3.944 \times 10^{-3}$ | $1.634 \times 10^{-3}$ | **0.03925%** | **1.05** | $2.00$ |
| **Analytical (Exact)**| 9.940903 | — | — | — | — | — |

### Numerical Stability & Spacetime Surface

<p align="center">
  <img src="figures/stability_demo.png" width="48%" alt="Von Neumann Stability Demo" />
  <img src="figures/heat_solution_surface.png" width="48%" alt="3D Spacetime Heat Solution" />
</p>

<p align="center">
  <img src="figures/prices_comparison.png" width="48%" alt="FDM Prices Comparison" />
  <img src="figures/error_analysis.png" width="48%" alt="Error and Convergence Orders" />
</p>

---

## Empirical Market Validation (^SPX Index Options)

### Why European ^SPX Options?

Real-world options validation must respect model assumptions:
* Single-stock options (e.g. `AAPL`) and ETF options (e.g. `SPY`) are **American-style**, granting early exercise rights. Using Black-Scholes on American options introduces non-negligible early exercise premium errors.
* **S&P 500 Index options (`^SPX`)** are strictly **European-style**, cash-settled at expiration with zero early exercise risk, making them the purest real-world instrument for testing Black-Scholes boundary conditions.

### Volatility Smile & Baseline Errors

We calibrated against **1,277 live European SPX options** across 3 maturities ($T = 0.08, 0.16, 0.41$ years):
* **FDM vs. Analytical Consistency:** FDM reproduces Black-Scholes analytical prices within $\$1.52$ ($< 0.1\%$) on actual market scales ($S = \$7,744.06$).
* **Variance Risk Premium:** Average ATM implied volatility is **$14.48\%$**, trading at an excess **$+1.81\%$** over 1-year historical realized volatility ($12.68\%$).
* **Volatility Skew:** Out-of-the-money crash puts trade up to **$42\%+$** implied volatility, yielding massive pricing errors under classical flat-$\sigma$ Black-Scholes (Baseline RMSE: **$\$35.14$**).

<p align="center">
  <img src="figures/market_volatility_smile.png" width="48%" alt="Market Volatility Smile" />
  <img src="figures/market_iv_surface.png" width="48%" alt="3D Implied Volatility Surface" />
</p>

---

## Literature Extensions & Progressive Error Relaxation

To eliminate the baseline $\$35.14$ pricing error, we systematically relaxed Black-Scholes assumptions using seminal quantitative finance literature:

1. **Leland (1985)** — *Discrete Hedging with Transaction Costs*:
   Replaces constant $\sigma$ with modified volatility $\tilde\sigma = \sigma \sqrt{1 + \text{sign}(\Gamma) \frac{c}{\sigma}\sqrt{\frac{2}{\pi \delta t}}}$, accounting for market friction ($c = 20\text{ bps}$, daily rebalancing $\delta t = 1/252$).
2. **Carr & Wu (2009)** — *Variance Risk Premium & ATM Calibration*:
   Replaces backward-looking historical volatility with market-implied forward ATM volatility $\sigma_{\text{ATM}}(T)$ per maturity.
3. **Merton (1976)** — *Jump-Diffusion Model*:
   Relaxes continuous paths by adding a Poisson crash jump process ($dq_t \sim \text{Poisson}(\lambda dt)$ with log-normal jump amplitude $\ln(1+J) \sim \mathcal{N}(\mu_J, \sigma_J^2)$):
   $$V_{\text{Merton}} = \sum_{n=0}^{\infty} \frac{e^{-\lambda' \tau} (\lambda' \tau)^n}{n!} V_{\text{BS}}\left(S, E, \tau, \sigma_n, r_n\right)$$
4. **Dupire (1994)** — *Local Volatility Model*:
   Extracts state-dependent local diffusion $\sigma_{\text{local}}(K, T)$ directly from the empirical strike and maturity smile surface.

### Relaxation Benchmark Results

Evaluated across all **1,277 traded European SPX options**:

| Model / Literature Extension | Assumption Relaxed | Overall RMSE ($) | Overall MAE ($) | RMSE Reduction |
| :--- | :--- | :---: | :---: | :---: |
| **0. Baseline Black-Scholes** | Classical model with Realized $\sigma = 12.68\%$ | **$35.14** | **$27.92** | *0.0% (Baseline)* |
| **1. Leland (1985)** | Frictionless continuous hedging ($c = 20\text{ bps}$, daily) | **$32.09** | **$26.28** | **-8.68%** |
| **2. ATM Calibrated (Carr-Wu 2009)** | Backward-looking historical $\sigma \to$ forward ATM term structure | **$31.82** | **$25.97** | **-9.43%** |
| **3. Dupire (1994) Local Volatility** | Constant $\sigma$ across strikes $\to$ local smile surface $\sigma(K, T)$ | **$17.84** | **$11.10** | **-49.23%** |
| **4. Merton (1976) Jump-Diffusion** | Continuous Gaussian paths $\to$ Poisson crash jumps ($\lambda = 0.58$, $\mu_J = -14.1\%$) | **$16.31** | **$10.82** | **-53.58%** |

### Model Progression Visualizations

<p align="center">
  <img src="figures/research_models_error_reduction.png" width="48%" alt="Error Reduction Across Literature Models" />
  <img src="figures/research_models_price_fit.png" width="48%" alt="Research Models Price Fit" />
</p>

### Economic Interpretation of Residual Error

Why does Merton (1976) leave a residual MAE of **$\$10.82$**?
* **Index Scale Context:** The underlying S&P 500 index spot is **$\$7,744.06$**; the mean option price in this dataset is **$\$264.76$**.
* An absolute error of $\$10.82$ represents just **$0.14\%$ of index spot** (equivalent to 14 cents on a $\$100$ equity option).
* In real market trading, the market maker's **bid-ask spread** across these deep SPX contracts ranges from **$\$3.00$ to $\$24.00$**.
* Therefore, the **$\$10.82$ residual error is completely contained within institutional market liquidity noise**.

---

## Installation & Quick Start

### Prerequisites
* Python 3.9+
* Recommended: Virtual environment (`venv` or `conda`)

### 1. Clone the Repository
```bash
git clone https://github.com/PraneelSatare/BSM-to-Market.git
cd BSM-to-Market
```

### 2. Set Up Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Pipeline Step-by-Step

```bash
# Step 1: Run numerical PDE solvers, stability checks, and convergence orders
python src/main.py

# Step 2: Fetch latest live European ^SPX option chain data (optional; pre-saved in data/)
python src/fetch_market_data.py

# Step 3: Run empirical European market validation and implied volatility smile extraction
python src/market_validation.py

# Step 4: Run literature model extensions (Leland, Carr-Wu, Merton, Dupire) and error relaxation
python src/model_extensions.py
```

All generated high-resolution diagnostics will be output directly into the `figures/` directory.

---

## References

1. **Zhu, Y.-l., Wu, X., Chern, I-L., & Sun, Z.-z.** (2013). *Derivative Securities and Difference Methods* (2nd ed.). Springer-Verlag.
2. **Black, F., & Scholes, M.** (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.
3. **Leland, H. E.** (1985). Option Pricing and Replication with Transactions Costs. *The Journal of Finance*, 40(5), 1283–1301.
4. **Merton, R. C.** (1976). Option Pricing When Underlying Stock Returns are Discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
5. **Dupire, B.** (1994). Pricing with a Smile. *Risk*, 7(1), 18–20.
6. **Carr, P., & Wu, L.** (2009). The Variance Risk Premium. *The Review of Financial Studies*, 22(3), 1311–1341.

---

## License

This project is licensed under the [MIT License](LICENSE).
