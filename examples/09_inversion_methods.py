#!/usr/bin/env python3
"""Comparison of optimisation methods for inversion.

Runs the same retrieval problem with L-BFGS-B, Nelder-Mead, and
differential evolution, comparing speed, accuracy, and convergence.

Set MCMC = True to include MCMC (slower).
"""

import time

import numpy as np

from biosnicar import run_model
from biosnicar.emulator import Emulator
from biosnicar.inverse import retrieve

PLOT = False
MCMC = False

# Load emulator and generate synthetic observation from the FULL FORWARD
# MODEL (not the emulator) so that the retrieval is honest.
emu = Emulator.load("data/emulators/glacier_ice_7_param_default.npz")

# Observing conditions (known, not retrieved)
fixed = {"solzen": 50, "direct": 1}

true_params = dict(rds=1000, rho=600, black_carbon=5000, glacier_algae=50000, dust=1000)
true_outputs = run_model(**true_params, **fixed, layer_type=1)
observed = np.array(true_outputs.albedo, dtype=np.float64)
parameters = ["rds", "rho", "black_carbon", "glacier_algae", "dust"]

print(f"True parameters: {true_params}\n")

# ======================================================================
# Compare gradient-based and derivative-free methods
# ======================================================================
methods = ["L-BFGS-B", "Nelder-Mead", "differential_evolution"]
results = {}

for method in methods:
    t0 = time.time()
    result = retrieve(
        observed=observed,
        parameters=parameters,
        emulator=emu,
        method=method,
        fixed_params=fixed,
    )
    elapsed = time.time() - t0
    results[method] = (result, elapsed)

    print(f"=== {method} ===")
    print(f"  Time:       {elapsed:.3f} s")
    print(f"  Converged:  {result.converged}")
    print(f"  Cost:       {result.cost:.6f}")
    print(f"  Func evals: {result.n_function_evals}")
    for name in parameters:
        err = abs(result.best_fit[name] - true_params[name])
        print(
            f"  {name:25s} = {result.best_fit[name]:10.1f}  "
            f"(err={err:.1f}, unc={result.uncertainty[name]:.1f})"
        )
    print()

# ======================================================================
# Summary table
# ======================================================================
print("=== Summary ===\n")
print(
    f"  {'Method':30s} {'Time (s)':>10s} {'Cost':>12s} {'Evals':>8s} {'rds err':>10s}"
)
print(f"  {'-' * 30} {'-' * 10} {'-' * 12} {'-' * 8} {'-' * 10}")
for method in methods:
    r, t = results[method]
    rds_err = abs(r.best_fit["rds"] - true_params["rds"])
    print(
        f"  {method:30s} {t:10.3f} {r.cost:12.6f} {r.n_function_evals:8d} {rds_err:10.1f}"
    )

# ======================================================================
# When to use each method
# ======================================================================
print("\n=== Guidance ===\n")
print("  L-BFGS-B:                Fast default. Uses gradients and box constraints.")
print("                           Best for well-constrained, smooth problems.")
print("  Nelder-Mead:             Derivative-free. More robust to noisy cost surfaces.")
print("                           Use when L-BFGS-B fails to converge.")
print("  differential_evolution:  Global search. Slower but explores full bounds.")
print(
    "                           Use when initial guess is poor or cost is multimodal."
)
print("  mcmc:                    Full Bayesian posterior. Use for publication-quality")
print("                           uncertainty or when parameters are degenerate.")

# ======================================================================
# MCMC (optional, slower)
# ======================================================================
if MCMC:
    print("\n=== MCMC (32 walkers, 1000 steps, 200 burn-in) ===\n")
    t0 = time.time()
    result_mcmc = retrieve(
        observed=observed,
        parameters=parameters,
        emulator=emu,
        method="mcmc",
        mcmc_walkers=32,
        mcmc_steps=1000,
        mcmc_burn=200,
        fixed_params=fixed,
    )
    elapsed = time.time() - t0
    print(f"  Time:       {elapsed:.1f} s")
    print(f"  Acceptance: {result_mcmc.acceptance_fraction:.3f}")
    print(f"  Chain shape: {result_mcmc.chains.shape}")
    for name in parameters:
        print(
            f"  {name:25s} = {result_mcmc.best_fit[name]:10.1f} "
            f"+/- {result_mcmc.uncertainty[name]:.1f}"
        )

    if PLOT:
        import matplotlib.pyplot as plt

        # Trace plot
        fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 2.5 * len(parameters)))
        for i, (ax, name) in enumerate(zip(axes, parameters)):
            ax.plot(result_mcmc.chains[:, :, i], alpha=0.3, linewidth=0.5)
            ax.axhline(true_params[name], color="r", linestyle="--", label="True")
            ax.set_ylabel(name)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Step")
        fig.suptitle("MCMC trace plots")
        fig.tight_layout()
        plt.show()

elif PLOT:
    import matplotlib.pyplot as plt

    wavelengths = np.arange(0.205, 4.999, 0.01)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavelengths, observed, "k-", label="Observed", linewidth=2)
    colors = ["r", "b", "g"]
    for (method, (result, _)), color in zip(results.items(), colors):
        ax.plot(
            wavelengths,
            result.predicted_albedo,
            "--",
            color=color,
            label=method,
            linewidth=1,
        )
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Albedo")
    ax.set_xlim(0.2, 2.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieved spectra from different optimisers")
    ax.legend()
    fig.tight_layout()
    plt.show()
