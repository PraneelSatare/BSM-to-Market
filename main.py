"""
Validate the Black-Scholes formula (2.90) by solving the transformed heat
equation (8.5) with finite differences.  Zhu et al., Derivative Securities and
Difference Methods, 2nd ed.
"""

import os
import warnings
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from bs_analytical import bs_call
from heat_transform import BSHeatTransform
from fdm_heat import HeatFDMSolver

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Book example (Sects. 2.6.5, 8.1.5, 8.2.1); book gives c(E,0)/E = 0.0994
E, T, r, D0, sigma = 100.0, 1.0, 0.10, 0.05, 0.20

# Wider than the book's [ln 0.2, ln 2.3]: its approximate upper BC pollutes S ≳ 150.
# x_l is a multiple of Δx so the payoff kink (x = 0) is a grid node.
x_l, x_u = -2.30, 1.60
M, N = 390, 500          # Δx = 0.01, α = 0.4

S_LO, S_HI = 50.0, 200.0


def quad_interp(xg, ug, xs):
    """3-point Lagrange interpolation (Sect. 8.1.1)."""
    j = int(np.clip(np.searchsorted(xg, xs), 1, len(xg) - 2))
    x0, x1, x2 = xg[j - 1], xg[j], xg[j + 1]
    u0, u1, u2 = ug[j - 1], ug[j], ug[j + 1]
    return (u0 * (xs - x1) * (xs - x2) / ((x0 - x1) * (x0 - x2))
            + u1 * (xs - x0) * (xs - x2) / ((x1 - x0) * (x1 - x2))
            + u2 * (xs - x0) * (xs - x1) / ((x2 - x0) * (x2 - x1)))


def make_solver(ht, Mi, Ni, xl=x_l, xu=x_u):
    return HeatFDMSolver(xl, xu, ht.tau_bar_max(), Mi, Ni,
                         f=ht.call_ic, f_l=ht.call_bc_lower,
                         f_u=lambda tb: ht.call_bc_upper(xu, tb))


def atm_price(ht, Mi, Ni, method):
    s = make_solver(ht, Mi, Ni)
    u_final = getattr(s, method)()[:, -1]
    return float(ht.u_to_V(quad_interp(s.x, u_final, ht.S_to_x(E, 0.0))))


def main():
    ht = BSHeatTransform(E, T, r, sigma, D0)
    tb_max = ht.tau_bar_max()
    solver = make_solver(ht, M, N)

    print("=" * 72)
    print("Black-Scholes validation via the heat equation  (Zhu et al., 2nd ed.)")
    print("=" * 72)
    print(f"  E = {E}, T = {T}, r = {r}, D0 = {D0}, σ = {sigma}")
    print(f"  τ̄_max = {tb_max:.4f},   r-D0-σ²/2 = {ht.mu:.4f}")
    print(f"  x ∈ [{x_l}, {x_u}]  (S at t=0 ∈ [{ht.x_to_S(x_l):.1f}, {ht.x_to_S(x_u):.1f}])")
    print(f"  M = {M}, N = {N},  Δx = {solver.dx:.4f},  Δτ̄ = {solver.dtau:.2e},"
          f"  α = {solver.alpha:.3f}")
    print("=" * 72)

    V_exact_atm = float(bs_call(E, E, T, r, sigma, D0))
    print(f"  Analytical (2.90):  c(E,0)/E = {V_exact_atm / E:.6f}   (book: 0.0994)")

    schemes = {
        "Explicit (7.8)": "solve_explicit",
        "Crank-Nicolson (7.9)": "solve_crank_nicolson",
        "Fully implicit*": "solve_implicit",
    }
    results = {name: getattr(solver, meth)() for name, meth in schemes.items()}

    S_grid = ht.x_to_S(solver.x, 0.0)
    V_fdm = {name: ht.u_to_V(u[:, -1], 0.0) for name, u in results.items()}
    V_exact = bs_call(S_grid, E, T, r, sigma, D0)
    roi = (S_grid >= S_LO) & (S_grid <= S_HI)
    xs_atm = ht.S_to_x(E, 0.0)

    print(f"\n  Errors over S ∈ [{S_LO:.0f}, {S_HI:.0f}] ({roi.sum()} nodes)\n")
    print(f"  {'Scheme':<22} {'V(E,0)':>10} {'Max |err|':>11} {'RMSE':>11} {'ATM rel (%)':>12}")
    print("  " + "-" * 70)
    for name in schemes:
        err = np.abs(V_fdm[name] - V_exact)[roi]
        V_atm = float(ht.u_to_V(quad_interp(solver.x, results[name][:, -1], xs_atm)))
        print(f"  {name:<22} {V_atm:10.6f} {err.max():11.3e} {np.sqrt(np.mean(err**2)):11.3e}"
              f" {abs(V_atm - V_exact_atm) / V_exact_atm * 100:12.5f}")
    print(f"  {'Analytical (2.90)':<22} {V_exact_atm:10.6f}")
    print("  * not in the book; shown for comparison.\n")

    styles = {"Explicit (7.8)": ("b--", 1.2), "Crank-Nicolson (7.9)": ("g:", 1.8),
              "Fully implicit*": ("r-.", 1.2)}

    # ---- Plot 1: prices ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S_grid[roi], V_exact[roi], "k-", lw=2, label="Analytical (2.90)")
    for name in schemes:
        ls, lw = styles[name]
        ax.plot(S_grid[roi], V_fdm[name][roi], ls, lw=lw, label=name)
    ax.axvline(E, color="grey", lw=0.8, alpha=0.6)
    ax.set_xlabel("Stock price  S")
    ax.set_ylabel("Call price  c(S, 0)")
    ax.set_title(f"European call — analytical vs FDM  (E={E:.0f}, r={r}, D0={D0}, σ={sigma}, T={T})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, "prices_comparison.png"), dpi=150)
    plt.close(fig)
    print("[✓] Saved prices_comparison.png")

    # ---- Plot 2: errors and convergence ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for name in schemes:
        ax.plot(S_grid[roi], np.abs(V_fdm[name] - V_exact)[roi], label=name)
    ax.set_xlabel("S")
    ax.set_ylabel("|V_fdm − V_exact|")
    ax.set_title(f"Absolute error  (M = {M}, N = {N}, α = {solver.alpha:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    rel_mask = roi & (V_exact > 0.01)
    for name in schemes:
        rel = np.abs(V_fdm[name] - V_exact)[rel_mask] / V_exact[rel_mask] * 100
        ax.semilogy(S_grid[rel_mask], rel, label=name)
    ax.set_xlabel("S")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Relative error  (where V > 0.01)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Time convergence: compare against fine-N on the same grid to isolate the time error
    print("  Running time-convergence study …")
    N_values = np.array([10, 20, 40, 80, 160, 320, 640])
    N_ref = 10000
    ref_cn = atm_price(ht, M, N_ref, "solve_crank_nicolson")
    ref_im = atm_price(ht, M, N_ref, "solve_implicit")
    err_cn = np.array([abs(atm_price(ht, M, n, "solve_crank_nicolson") - ref_cn) for n in N_values])
    err_im = np.array([abs(atm_price(ht, M, n, "solve_implicit") - ref_im) for n in N_values])

    ax = axes[1, 0]
    ax.loglog(N_values, err_im, "o-", color="tab:red", label="Fully implicit*")
    ax.loglog(N_values, err_cn, "s-", color="tab:green", label="Crank-Nicolson (7.9)")
    ax.loglog(N_values, err_im[-1] * (N_values[-1] / N_values), "k:", lw=1, label="O(Δτ̄)  slope −1")
    ax.loglog(N_values, err_cn[-1] * (N_values[-1] / N_values) ** 2, "k--", lw=1, label="O(Δτ̄²) slope −2")
    ax.set_xlabel("N  (time steps)")
    ax.set_ylabel("ATM error vs fine-N reference")
    ax.set_title(f"Convergence in time  (M = {M} fixed)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Space convergence: Δx halves each time and x = 0 stays a node
    print("  Running space-convergence study …")
    M_values = np.array([78, 156, 312, 624, 1248])
    N_s = 4000
    err_M = np.array([abs(atm_price(ht, m, N_s, "solve_crank_nicolson") - V_exact_atm)
                      for m in M_values])

    ax = axes[1, 1]
    ax.loglog(M_values, err_M, "s-", color="tab:orange", label="Crank-Nicolson (7.9)")
    ax.loglog(M_values, err_M[0] * (M_values[0] / M_values) ** 2, "k--", lw=1, label="O(Δx²) slope −2")
    ax.set_xlabel("M  (space steps)")
    ax.set_ylabel("ATM error vs analytical (2.90)")
    ax.set_title(f"Convergence in space  (N = {N_s} fixed)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Error analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(SCRIPT_DIR, "error_analysis.png"), dpi=150)
    plt.close(fig)
    print("[✓] Saved error_analysis.png")
    print(f"      observed order in time : implicit ≈ {np.log2(err_im[-2] / err_im[-1]):.2f},"
          f"  Crank-Nicolson ≈ {np.log2(err_cn[-2] / err_cn[-1]):.2f}")
    print("      observed order in space: Crank-Nicolson ≈ "
          + ", ".join(f"{np.log2(err_M[i] / err_M[i + 1]):.2f}" for i in range(len(M_values) - 1)))

    # ---- Plot 3: u(x, τ̄) surface ----
    X, TB = np.meshgrid(solver.x, solver.tau, indexing="ij")
    u_cn = results["Crank-Nicolson (7.9)"]
    sx, st = max(1, M // 80), max(1, N // 80)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X[::sx, ::st], TB[::sx, ::st], u_cn[::sx, ::st],
                    cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("τ̄")
    ax.set_zlabel("u(x, τ̄)")
    ax.set_title("Heat-equation solution u(x, τ̄) — Crank-Nicolson (7.9)")
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, "heat_solution_surface.png"), dpi=150)
    plt.close(fig)
    print("[✓] Saved heat_solution_surface.png")

    # ---- Plot 4: explicit stability, condition (8.7) ----
    # Coarser grid so the unstable run has enough steps for errors to grow visibly.
    M_demo = 195                                        # Δx = 0.02
    dx_demo = (x_u - x_l) / M_demo
    N_crit = int(np.ceil(tb_max / (0.5 * dx_demo ** 2)))
    N_unstable = int(0.7 * N_crit)
    N_stable = 2 * N_crit

    s_un = make_solver(ht, M_demo, N_unstable)
    s_st = make_solver(ht, M_demo, N_stable)
    u_un = s_un.solve_explicit()[:, -1]
    u_st = s_st.solve_explicit()[:, -1]
    u_ex = ht.V_to_u(bs_call(ht.x_to_S(s_st.x), E, T, r, sigma, D0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(s_un.x, u_un, "r-", lw=0.8, label="Explicit (8.6)")
    ax1.plot(s_st.x, u_ex, "k--", lw=1.5, label="Analytical u")
    ax1.set_title(f"UNSTABLE: α = {s_un.alpha:.3f} > 1/2   (M={M_demo}, N={N_unstable})")
    ax2.plot(s_st.x, u_st, "b-", lw=1.2, label="Explicit (8.6)")
    ax2.plot(s_st.x, u_ex, "k--", lw=1.5, label="Analytical u")
    ax2.set_title(f"STABLE: α = {s_st.alpha:.3f} ≤ 1/2   (M={M_demo}, N={N_stable})")
    for ax in (ax1, ax2):
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, τ̄_max)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Explicit scheme stability, condition (8.7)  α = Δτ̄/Δx² ≤ 1/2",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(SCRIPT_DIR, "stability_demo.png"), dpi=150)
    plt.close(fig)
    print(f"[✓] Saved stability_demo.png  (unstable α={s_un.alpha:.3f}, N={N_unstable};"
          f" stable α={s_st.alpha:.3f}, N={N_stable})")

    # ---- Plot 5: simultaneous (M, N) refinement with constant α = 0.40 ----
    print("  Running simultaneous (M, N) study with constant α = 0.40 …")
    alpha_const = 0.40
    M_const = np.array([78, 156, 312, 624, 1248])
    N_const = np.array([int(round(tb_max / (alpha_const * ((x_u - x_l) / m) ** 2))) for m in M_const])

    err_exp_const, err_cn_const, err_im_const = [], [], []
    spatial_profiles = []

    for m, n in zip(M_const, N_const):
        s = HeatFDMSolver(x_l, x_u, tb_max, m, n,
                          f=ht.call_ic, f_l=ht.call_bc_lower,
                          f_u=lambda tb: ht.call_bc_upper(x_u, tb))
        u_exp = s.solve_explicit()[:, -1]
        u_cn = s.solve_crank_nicolson()[:, -1]
        u_im = s.solve_implicit()[:, -1]

        v_exp_atm = float(ht.u_to_V(quad_interp(s.x, u_exp, xs_atm)))
        v_cn_atm = float(ht.u_to_V(quad_interp(s.x, u_cn, xs_atm)))
        v_im_atm = float(ht.u_to_V(quad_interp(s.x, u_im, xs_atm)))

        err_exp_const.append(abs(v_exp_atm - V_exact_atm))
        err_cn_const.append(abs(v_cn_atm - V_exact_atm))
        err_im_const.append(abs(v_im_atm - V_exact_atm))

        S_nodes = ht.x_to_S(s.x, 0.0)
        V_exact_nodes = bs_call(S_nodes, E, T, r, sigma, D0)
        V_cn_nodes = ht.u_to_V(u_cn, 0.0)
        roi_nodes = (S_nodes >= S_LO) & (S_nodes <= S_HI)
        spatial_profiles.append((S_nodes[roi_nodes], np.abs(V_cn_nodes - V_exact_nodes)[roi_nodes], m, n))

    err_exp_const = np.array(err_exp_const)
    err_cn_const = np.array(err_cn_const)
    err_im_const = np.array(err_im_const)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Convergence across all three schemes
    ax1.loglog(M_const, err_exp_const, "o-", color="tab:blue", label="Explicit (7.8)")
    ax1.loglog(M_const, err_cn_const, "s-", color="tab:green", label="Crank-Nicolson (7.9)")
    ax1.loglog(M_const, err_im_const, "^-", color="tab:red", label="Fully implicit*")
    ax1.loglog(M_const, err_cn_const[0] * (M_const[0] / M_const) ** 2, "k--", lw=1.2,
               label=r"$\mathcal{O}(\Delta x^2) \sim M^{-2}$ slope −2")
    ax1.set_xlabel(r"Spatial steps $M$  ($N \propto M^2$, preserving $\alpha = 0.40$)")
    ax1.set_ylabel(r"ATM absolute error $|V(E, 0) - V_{\mathrm{exact}}|$")
    ax1.set_title(rf"ATM Error Convergence along $\alpha = {alpha_const:.2f}$ (const)")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # Subplot 2: Spatial distribution of error as grid is refined
    for S_pts, err_pts, m, n in spatial_profiles:
        ax2.plot(S_pts, err_pts, label=f"M={m}, N={n}")
    ax2.axvline(E, color="grey", lw=0.8, linestyle=":", alpha=0.7, label=f"Strike E={E:.0f}")
    ax2.set_xlabel("Stock price S")
    ax2.set_ylabel(r"$|V_{\mathrm{CN}} - V_{\mathrm{exact}}|$")
    ax2.set_title(rf"Crank-Nicolson Spatial Error Profiles ($\alpha = {alpha_const:.2f}$)")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(rf"Simultaneous $(M, N)$ Refinement at Constant Mesh Ratio $\alpha = \Delta\bar\tau / \Delta x^2 = {alpha_const:.2f}$",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(SCRIPT_DIR, "constant_alpha_analysis.png"), dpi=150)
    plt.close(fig)
    print("[✓] Saved constant_alpha_analysis.png")
    print(f"      observed order at constant α: Explicit ≈ {np.log2(err_exp_const[-2] / err_exp_const[-1]):.2f},"
          f"  Crank-Nicolson ≈ {np.log2(err_cn_const[-2] / err_cn_const[-1]):.2f},"
          f"  Fully implicit ≈ {np.log2(err_im_const[-2] / err_im_const[-1]):.2f}")
    print("\nDone.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Explicit scheme")   # expected in the stability demo
    main()
