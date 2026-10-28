# Black-Scholes PDE Validation via the Heat Equation

Follows Zhu, Wu, Chern & Sun, *Derivative Securities and Difference Methods*, 2nd ed. (Springer, 2013). Equation and section numbers refer to this book.

| Book | Content | File |
|:---|:---|:---|
| §2.6.1, (2.73)–(2.78) | Black-Scholes → heat equation | `heat_transform.py` |
| §2.6.5, (2.90)–(2.92) | Black-Scholes formulae | `bs_analytical.py` |
| §7.1, (7.8)–(7.10); §7.2.1 | Explicit and Crank-Nicolson schemes; stability | `fdm_heat.py` |
| §8.1.1, (8.5)–(8.7); §8.2.1 | Pricing with $u, x, \bar\tau$; boundary conditions | `main.py`, `heat_transform.py` |

## 1. Problem (2.73)

$$
V_t + \tfrac12\sigma^2 S^2 V_{SS} + (r - D_0) S V_S - rV = 0, \qquad V(S,T) = \max(S - E, 0),
$$
with $S$ and $V$ divided by the exercise price $E$.

## 2. Transformation (2.78)

$$
x = \ln S + \big(r - D_0 - \tfrac12\sigma^2\big)(T-t), \qquad
\bar\tau = \tfrac12\sigma^2 (T-t), \qquad
V = e^{-r(T-t)}\, u(x,\bar\tau),
$$
which gives (2.77)/(8.5):
$$
u_{\bar\tau} = u_{xx}, \qquad u(x,0) = \max(e^x - 1, 0).
$$

Boundary conditions on $[x_l, x_u]$ (§8.2.1), with $k = 2(r-D_0)/\sigma^2$:
$$
u(x_l,\bar\tau) = 0, \qquad u(x_u,\bar\tau) = \big(e^{x_u - (k-1)\bar\tau} - 1\big) e^{k\bar\tau}.
$$

## 3. Schemes (§7.1)

Grid: $x_m = x_l + m\Delta x$, $\bar\tau_n = n\Delta\bar\tau$, mesh ratio $\alpha = \Delta\bar\tau/\Delta x^2$.

**Explicit (7.8):** $u_m^{n+1} = \alpha u_{m+1}^n + (1 - 2\alpha)u_m^n + \alpha u_{m-1}^n$.
Error $O(\Delta x^2, \Delta\bar\tau)$. Stable iff $\alpha \le \tfrac12$, since the eigenvalues are $1 - 4\alpha\sin^2(\omega_k/2)$ (§7.2.1).

**Crank-Nicolson (7.9)/(7.10):** $A\mathbf u^{n+1} = B\mathbf u^n + \mathbf b^n$, where $A = \mathrm{tridiag}(-\tfrac\alpha2, 1+\alpha, -\tfrac\alpha2)$ and $B = \mathrm{tridiag}(\tfrac\alpha2, 1-\alpha, \tfrac\alpha2)$.
Error $O(\Delta x^2, \Delta\bar\tau^2)$. Unconditionally stable.

**Fully implicit:** not in the book; included for comparison only. First order in $\bar\tau$.

## 4. Recovering prices (§8.1.1)

$$
V(S,0) = E\, e^{-rT}\, u\big(\ln(S/E) + (r - D_0 - \tfrac12\sigma^2)T,\ \tfrac12\sigma^2T\big),
$$
using quadratic interpolation between grid points.

## 5. Parameters

These are the book's example values: $E = 100$, $T = 1$, $r = 0.1$, $D_0 = 0.05$, $\sigma = 0.2$. The book gives $c/E = 0.0994$ and $p/E = 0.0530$; `bs_analytical.py` reproduces both.

Grid: $x \in [-2.3, 1.6]$ ($S/E \approx 0.1$–$5$), $M = 390$ ($\Delta x = 0.01$), $N = 500$ ($\alpha = 0.4$).

This domain is wider than the book's suggested $[\ln 0.2, \ln 2.3]$, because the approximate upper boundary condition distorts prices for $S \gtrsim 150$ on the narrower domain. $x_l$ is a multiple of $\Delta x$, so the payoff kink at $x = 0$ falls on a grid node.

## 6. Run

```bash
cd "Documents/Sem 7/UGP"
python3 main.py
```

## 7. Results

```
Scheme                     V(E,0)   Max |err|        RMSE  ATM rel (%)
----------------------------------------------------------------------
Explicit (7.8)           9.940409   1.104e-03   6.540e-04      0.00496
Crank-Nicolson (7.9)     9.938706   2.220e-03   9.684e-04      0.02210
Fully implicit*          9.937001   3.944e-03   1.634e-03      0.03925
Analytical (2.90)        9.940903

observed order in time : implicit ≈ 1.05,  Crank-Nicolson ≈ 2.00
observed order in space: Crank-Nicolson ≈ 2.02, 2.00, 2.00, 2.00
observed order at constant α: Explicit ≈ 2.00,  Crank-Nicolson ≈ 2.00,  Fully implicit ≈ 2.00
```

| Plot | Shows |
|:---|:---|
| `prices_comparison.png` | FDM prices on top of the analytical curve |
| `error_analysis.png` | Error vs $S$; convergence in time (Crank-Nicolson slope −2, implicit −1) and in space (slope −2) |
| `heat_solution_surface.png` | $u(x,\bar\tau)$ from Crank-Nicolson |
| `stability_demo.png` | Explicit scheme blows up at $\alpha = 0.714$ and is stable at $\alpha = 0.248$ (cf. Fig. 8.6) |
| `constant_alpha_analysis.png` | Simultaneous $(M, N)$ refinement along constant $\alpha = 0.40$ showing $O(\Delta x^2)$ convergence for all schemes and spatial error decay |

## 8. Real Market Validation (European ^SPX Options)

We benchmarked the Black-Scholes analytical formula and Crank-Nicolson FDM solver against **1,277 live, traded European-style option quotes** on the **S&P 500 Index (^SPX)** across three monthly maturities ($T \in [0.08, 0.25]$ years), with index spot $S = \$7,744.06$, risk-free rate $r = 4.09\%$, and dividend yield $D_0 = 1.25\%$.

Unlike SPY options (which are American style with early-exercise risk), **SPX options are cash-settled and strictly European**, matching the exact boundary conditions of the Black-Scholes PDE.

### Scripts
1. **`fetch_market_data.py`**: Downloads 1-year daily index prices, estimates realized volatility ($\sigma_{\text{realized}} = 12.68\%$), fetches European option chains, and saves to `spx_options_market_data.csv`.
2. **`market_validation.py`**: Inverts European Black-Scholes to extract Implied Volatilities (Brent's root-finding), calculates pricing errors against realized volatility, tests Crank-Nicolson FDM on market parameters, and outputs error diagnostics.

```bash
# Fetch European market data and run validation:
.ugp/bin/python3 fetch_market_data.py
.ugp/bin/python3 market_validation.py
```

### Empirical Findings
* **Crank-Nicolson FDM Consistency**: On a $\$7,744$ index, the FDM price matches the Black-Scholes analytical price within $\$1.52$ ($< 0.1\%$ relative difference), confirming that the numerical solver handles real market scales accurately.
* **Variance Risk Premium (VRP)**: Average At-The-Money (ATM) Implied Volatility is **$14.48\%$**, trading at a **$+1.81\%$** premium above historical realized volatility ($12.68\%$), proving that even European options trade at an implied volatility premium over historical volatility.
* **Empirical Volatility Skew**: Implied volatility rises smoothly from $12\%$ to over $42\%$ for Out-Of-The-Money crash-protection puts ($K/S < 0.90$), illustrating that the constant-volatility assumption fails even in strictly European markets.

| Market Plot | Shows |
|:---|:---|
| `market_volatility_smile.png` | Empirical Volatility Skew ($\sigma_{\text{imp}}$ vs Moneyness $K/S$) across maturities vs flat realized $\sigma$ |
| `market_pricing_errors.png` | Actual Market Prices vs Black-Scholes model prices and pricing error bias vs Moneyness |
| `market_iv_surface.png` | 3D Implied Volatility Surface $\sigma_{\text{imp}}(K/S, T)$ across strike and maturity |

## 9. Research Paper Extensions & Progressive Error Reduction

We systematically relaxed Black-Scholes assumptions using literature models on the **1,277 live European SPX options** to evaluate progressive error reduction:

| Model / Literature Extension | Assumption Relaxed | RMSE ($) | MAE ($) | RMSE Reduction |
| :--- | :--- | :---: | :---: | :---: |
| **0. Baseline Black-Scholes** | Classical model with Realized $\sigma = 12.68\%$ | **$35.14** | **$27.92** | *0.0% (Baseline)* |
| **1. Leland (1985)** | Frictionless continuous hedging ($c = 20\text{ bps}$, daily) | **$32.09** | **$26.28** | **-8.68%** |
| **2. ATM Calibrated (Carr-Wu 2009)** | Backward-looking historical $\sigma$ $\to$ forward ATM term structure | **$31.82** | **$25.97** | **-9.43%** |
| **3. Merton (1976) Jump-Diffusion** | Continuous Gaussian paths $\to$ Poisson crash jumps ($\lambda = 0.58$, $\mu_J = -14.1\%$) | **$16.31** | **$10.82** | **-53.58%** |
| **4. Dupire (1994) Local Volatility** | Constant $\sigma$ across strikes $\to$ local smile surface $\sigma(K, T)$ | **$17.84** | **$11.10** | **-49.23%** |

```bash
# Run the progressive error reduction benchmark:
.ugp/bin/python3 model_extensions.py
```

| Literature Extension Plot | Shows |
|:---|:---|
| `research_models_error_reduction.png` | Bar chart of progressive RMSE & MAE reduction from Baseline down to Dupire |
| `research_models_price_fit.png` | Actual Market Prices vs Model fit across strikes and elimination of put skew errors |

## References

1. Y.-l. Zhu, X. Wu, I-L. Chern, Z.-z. Sun, *Derivative Securities and Difference Methods*, 2nd ed., Springer, 2013.
2. F. Black, M. Scholes, *J. Political Economy* 81(3), 637–654, 1973.
3. H. E. Leland, *J. Finance* 40(5), 1283–1301, 1985.
4. R. C. Merton, *J. Financial Economics* 3(1–2), 125–144, 1976.
5. B. Dupire, *Risk* 7(1), 18–20, 1994.
6. P. Carr, L. Wu, *Review of Financial Studies* 22(3), 1311–1341, 2009.
