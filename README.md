# Black-Scholes to Market: PDE Finite Difference Solvers, Empirical SPX Calibration, & Literature Error Relaxation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Quantitative Finance](https://img.shields.io/badge/Field-Quantitative%20Finance-emerald.svg)](#)
[![PDE Solvers](https://img.shields.io/badge/Numerics-FDM%20%7C%20Crank--Nicolson-orange.svg)](#)

A quantitative finance framework connecting mathematical PDE theory to real-world options markets:
1. Analytical **Black-Scholes-Merton (1973)** PDE and its transformation into the **1D Heat Diffusion Equation** (Zhu et al., 2013).
2. Numerical **Finite Difference Methods (FDM)** — Explicit FTCS, Crank-Nicolson, and Fully Implicit BTCS — with von Neumann stability and spacetime convergence verification ($\mathcal{O}(\Delta x^2), \mathcal{O}(\Delta\tau^2)$).
3. **Empirical European Market Calibration** on **1,337 live S&P 500 (`^SPX`)** index options across 3 maturities, utilizing put-call parity to infer exact forward prices and eliminate artificial call/put implied volatility discrepancies.
4. **Progressive Error Relaxation** using literature models (**Leland 1985**, **ATM Calibration**, **Smile-Interpolated BS**, and **Merton 1976 Jump-Diffusion**), evaluated with both in-sample metrics and 5-fold cross-validation out-of-sample testing.

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
  - [Put-Call Parity Forward Inference & Volatility Smile](#put-call-parity-forward-inference--volatility-smile)
  - [Diffusion-Scaled FDM Grid Consistency](#diffusion-scaled-fdm-grid-consistency)
- [Literature Extensions & Progressive Error Relaxation](#literature-extensions--progressive-error-relaxation)
  - [Models & Assumptions Relaxed](#models--assumptions-relaxed)
  - [Benchmark Results (In-Sample & 5-Fold Cross-Validation)](#benchmark-results-in-sample--5-fold-cross-validation)
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
│   ├── iv_utils.py               # Put-call parity forward solver, Brent's IV inverter, vectorized pricer
│   ├── main.py                   # PDE validation, stability, and convergence studies
│   ├── fetch_market_data.py      # European ^SPX option chain & historical price downloader
│   ├── market_validation.py      # Empirical European market benchmarking & IV surface analysis
│   └── model_extensions.py       # Literature models (Leland, ATM VRP, Smile-Interp, Merton Jumps)
│
├── data/                         # Real-world market datasets
│   ├── spx_options_market_data.csv # 1,337 traded European SPX option quotes across 3 maturities
│   └── spx_underlying_history.csv  # 1-year historical daily closes for S&P 500 index
│
├── figures/                      # High-resolution generated plots and diagnostic figures
│   ├── prices_comparison.png     # FDM curves overlaid on analytical Black-Scholes
│   ├── error_analysis.png        # Time and space convergence rates (orders O(Δτ²) and O(Δx²))
│   ├── heat_solution_surface.png # 3D spacetime heat solution surface u(x, τ̄)
│   ├── stability_demo.png        # Von Neumann numerical stability criterion (α ≤ 1/2)
│   ├── constant_alpha_analysis.png # Simultaneous (M, N) refinement at constant α = 0.40
│   ├── market_volatility_smile.png # Empirical European SPX volatility skew from OTM options
│   ├── market_pricing_errors.png # European market vs. constant-σ pricing errors across moneyness
│   ├── market_iv_surface.png     # 3D Implied Volatility Surface σ_imp(K/F, T) for ^SPX
│   ├── research_models_error_reduction.png # Progressive RMSE/MAE drops across literature models
│   └── research_models_price_fit.png       # Fit comparison across strikes & elimination of put skew error
│
├── .gitignore                    # Python and cache exclusions
├── LICENSE                       # MIT License
├── requirements.txt              # Dependencies (numpy, scipy, matplotlib, pandas, yfinance)
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
   *Stability:* Conditionally stable iff $\alpha \le \frac{1}{2}$ (Von Neumann criterion).

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
* Single-stock options (e.g. `AAPL`) and ETF options (e.g. `SPY`) are **American-style**, granting early exercise rights. Using Black-Scholes on American options introduces early-exercise boundary distortion.
* **S&P 500 Index options (`^SPX`)** are strictly **European-style**, cash-settled at expiration with zero early exercise risk, matching the terminal payoff condition of the Black-Scholes PDE.

### Put-Call Parity Forward Inference & Volatility Smile

Rather than assuming a static dividend yield, we infer the market forward price $F_{\mathrm{PCP}}(T)$ and implied dividend yield $q(T)$ directly from put-call parity on overlapping strikes:

$$C(K) - P(K) = [F_{\mathrm{PCP}} - K] e^{-r T} \implies F_{\mathrm{PCP}} = [C(K) - P(K)] e^{r T} + K$$

* **Calibrated Forwards:** Across the 3 maturities ($T = 0.085\text{y}, 0.172\text{y}, 0.249\text{y}$), the empirical forward prices are $\$7,775.10$, $\$7,797.93$, and $\$7,827.36$.
* **Elimination of Call/Put Discrepancy:** With the PCP forward, ATM call and put implied volatilities agree to within **$0.004\%$** ($< 0.5\text{ bps}$), eliminating the $1.5\text{--}2.0\text{ vol-point}$ gap caused by misspecified yields.
* **Liquid OTM Framing:** Implied volatility is constructed using pure out-of-the-money options (puts for $K \le F$, calls for $K > F$), covering **774 liquid contracts** with **100% successful IV extraction**.
* **Empirical Skew:** Out-of-the-money put implied volatility rises from $13.2\%$ ATM up to **$39.59\%$** at $K/F = 0.77$, reflecting institutional downside tail demand.

<p align="center">
  <img src="figures/market_volatility_smile.png" width="48%" alt="Market Volatility Smile" />
  <img src="figures/market_iv_surface.png" width="48%" alt="3D Implied Volatility Surface" />
</p>

### Diffusion-Scaled FDM Grid Consistency

For short maturities ($T = 0.085\text{y}$, $\sigma = 12.68\%$), the dimensionless heat horizon is small: $\bar\tau_{\max} = \frac{1}{2}\sigma^2 T \approx 0.00068$, corresponding to a diffusion length scale $\sigma\sqrt{T} \approx 0.0369$.
* Using an unscaled spatial grid ($[-2.5, 2.5]$ with $M=400$) yields $\Delta x = 0.0125$, leaving only $\sim 3$ grid cells across the entire diffusion layer and inducing artificial truncation error.
* By scaling the spatial domain to the physical diffusion length ($x \in [-8\sigma\sqrt{T}, +8\sigma\sqrt{T}] \approx [-0.30, 0.30]$ with $M=600, N=500$), the grid spacing is refined to $\Delta x = 0.0010$.
* **Benchmark Result:** On the ATM contract ($S = \$7,744.06, E = \$7,750, T = 0.0849\text{y}$):
  $$\text{Analytical BS Price} = \$126.9080, \quad \text{Crank-Nicolson FDM Price} = \$126.8976, \quad \text{Error} = \mathbf{\$0.0104} \ (\mathbf{0.0082\%})$$
  confirming that the FDM solver matches the analytical Black-Scholes solution to within a fraction of a cent.

---

## Literature Extensions & Progressive Error Relaxation

### Models & Assumptions Relaxed

1. **Leland (1985)** — *Discrete Hedging with Transaction Costs*:
   Relaxes frictionless continuous hedging ($dt \to 0$). Discrete daily rebalancing ($\delta t = 1/252$) with proportional transaction costs ($c = 20\text{ bps}$) widens effective replication volatility:
   $$\tilde\sigma = \sigma \sqrt{1 + \text{sign}(\Gamma) \frac{c}{\sigma}\sqrt{\frac{2}{\pi \delta t}}} = 13.89\%$$
2. **ATM Implied Volatility Calibration**:
   Relaxes the backward-looking historical volatility assumption ($\sigma_{\mathrm{real}} = 12.68\%$), calibrating to forward ATM market implied volatility ($12.44\% - 13.46\%$) to absorb the market's Variance Risk Premium.
3. **Smile-Interpolated Black-Scholes**:
   Relaxes strike-invariance of volatility by fitting a per-expiry polynomial smile $\sigma_{\mathrm{fit}}(\ln(K/F))$ to the empirical OTM market smile and evaluating via Black-Scholes.
4. **Merton (1976)** — *Jump-Diffusion Model*:
   Relaxes continuous Gaussian paths by adding a Poisson crash jump process:
   $$\frac{dS_t}{S_{t^-}} = (r - q - \lambda \kappa) dt + \sigma_{\text{diff}} dW_t + J \, dq_t, \qquad dq_t \sim \text{Poisson}(\lambda dt), \ \ln(1+J) \sim \mathcal{N}(\mu_J, \delta_J^2)$$
   Calibrated structural parameters: $\lambda = 0.53\text{ jumps/yr}$, mean jump $\mu_J = -15.6\%$, jump spread $\delta_J = 10.9\%$, continuous volatility $\sigma_{\text{diff}} = 9.42\%$.

### Benchmark Results (In-Sample & 5-Fold Cross-Validation)

Evaluated across **774 traded liquid OTM European SPX options**:

| Model / Literature Extension | Parameters / Degrees of Freedom | In-Sample RMSE ($) | In-Sample MAE ($) | RMSE Drop (%) | 5-Fold CV Out-of-Sample RMSE ($) | Out-of-Sample MAE ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Baseline Black-Scholes** | 0 (Historical Realized $\sigma = 12.68\%$) | **$27.14** | **$22.45** | *Baseline (0.0%)* | — | — |
| **1. Leland (1985)** | 0 ($c = 20\text{ bps}$, daily $\delta t$) | **$24.31** | **$21.52** | **-10.42%** | — | — |
| **2. ATM Calibrated** | 1 per expiry ($\sigma_{\mathrm{ATM}}(T)$) | **$24.97** | **$21.40** | **-7.99%** | — | — |
| **3. Smile-Interpolated BS** | 3 per expiry (quadratic in $\ln(K/F)$) | **$5.65** | **$4.55** | **-79.19%** | **$5.69** | **$4.59** |
| **4. Merton (1976) Jumps** | 4 structural ($\lambda, \mu_J, \delta_J, \sigma_{\text{diff}}$) | **$3.08** | **$2.35** | **-88.64%** | **$3.10** | **$2.37** |

<p align="center">
  <img src="figures/research_models_error_reduction.png" width="48%" alt="Error Reduction Across Literature Models" />
  <img src="figures/research_models_price_fit.png" width="48%" alt="Research Models Price Fit" />
</p>

### Economic Interpretation of Residual Error

* **Residual Scale:** Under Merton (1976), the residual MAE is **$\$2.35$** on an underlying index of **$\$7,744.06$** (equivalent to 3 cents on a $\$100$ equity option).
* **Comparison to Bid-Ask Spread:** In the traded SPX dataset, the **median bid-ask spread is $\$0.80$** for OTM options and **$\$1.30$** across all options (with a mean spread of **$\$3.32$**).
* **Conclusion:** While baseline constant-$\sigma$ Black-Scholes exhibits errors $17\times\text{--}28\times$ larger than market friction, the Merton jump-diffusion model reduces pricing error to **within institutional bid-ask market liquidity noise**.

---

## Installation & Quick Start

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

# Step 4: Run literature model extensions, cross-validation, and error relaxation
python src/model_extensions.py
```

All diagnostic plots are saved to the `figures/` directory.

---

## References

1. **Zhu, Y.-l., Wu, X., Chern, I-L., & Sun, Z.-z.** (2013). *Derivative Securities and Difference Methods* (2nd ed.). Springer-Verlag.
2. **Black, F., & Scholes, M.** (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.
3. **Leland, H. E.** (1985). Option Pricing and Replication with Transactions Costs. *The Journal of Finance*, 40(5), 1283–1301.
4. **Merton, R. C.** (1976). Option Pricing When Underlying Stock Returns are Discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
5. **Carr, P., & Wu, L.** (2009). The Variance Risk Premium. *The Review of Financial Studies*, 22(3), 1311–1341.

---

## License

This project is licensed under the [MIT License](LICENSE).
