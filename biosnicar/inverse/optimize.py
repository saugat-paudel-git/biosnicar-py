"""Retrieval dispatcher for spectral and band-mode inversion.

Provides :func:`retrieve`, the main entry point for estimating ice
physical properties from observed albedo using an emulator or direct
forward model.

Supported optimisation methods:

* ``"L-BFGS-B"`` — fast quasi-Newton (default)
* ``"Nelder-Mead"`` — derivative-free simplex
* ``"differential_evolution"`` — global search
* ``"mcmc"`` — Bayesian posterior sampling via emcee
"""

import warnings

import numpy as np
from scipy.optimize import minimize, differential_evolution

from biosnicar.inverse.cost import spectral_cost, band_cost
from biosnicar.inverse.result import RetrievalResult


# Default parameter bounds for glacier ice retrieval
DEFAULT_BOUNDS = {
    "rds": (100.0, 5000.0),
    "rho": (100.0, 917.0),
    "solzen": (20.0, 80.0),
    "direct": (0, 1),
    "black_carbon": (0.0, 100000.0),
    "snow_algae": (0.0, 500000.0),
    "glacier_algae": (0.0, 100000.0),
}

DEFAULT_X0 = {
    "rds": 1000.0,
    "rho": 500.0,
    "solzen": 50.0,
    "direct": 1,
    "black_carbon": 100.0,
    "snow_algae": 10000.0,
    "glacier_algae": 100.0,
}


def retrieve(
    observed,
    parameters,
    emulator=None,
    forward_fn=None,
    platform=None,
    observed_band_names=None,
    obs_uncertainty=None,
    bounds=None,
    x0=None,
    regularization=None,
    wavelength_mask=None,
    method="L-BFGS-B",
    mcmc_walkers=32,
    mcmc_steps=2000,
    mcmc_burn=500,
    fixed_params=None,
):
    """Retrieve ice physical properties from observed albedo.

    Supports two observation modes:

    * **Spectral mode** (default): *observed* is a 480-element array of
      spectral albedo.  Compared directly to the emulator/forward-model
      prediction.
    * **Band mode**: *observed* is an N-element array of satellite band
      albedo values.  Set *platform* and *observed_band_names* to
      activate this mode.

    And two forward-model modes:

    * **Emulator mode** (recommended): pass an :class:`Emulator` instance.
      ~microsecond evaluations enable fast optimisation and MCMC.
    * **Direct mode**: pass a callable *forward_fn* that returns a
      480-element albedo array.  Slower but exact.

    Parameters
    ----------
    observed : np.ndarray
        Observed albedo — 480-band spectral array or N-band satellite array.
    parameters : list of str
        Names of parameters to retrieve (e.g.
        ``["rds", "black_carbon"]``).
    emulator : Emulator, optional
        Trained emulator for fast forward evaluation.
    forward_fn : callable, optional
        Direct forward function: ``forward_fn(**params) -> np.ndarray (480,)``.
        Either *emulator* or *forward_fn* must be provided.
    platform : str, optional
        Satellite platform key (e.g. ``"sentinel2"``).  Activates band mode.
    observed_band_names : list of str, optional
        Band names matching *observed* elements (e.g.
        ``["B3", "B4", "B11"]``).  Required when *platform* is set.
    obs_uncertainty : np.ndarray, optional
        Per-element 1-sigma measurement uncertainty.
    bounds : dict, optional
        ``{param_name: (lo, hi)}``.  Overrides :data:`DEFAULT_BOUNDS`.
    x0 : dict, optional
        ``{param_name: value}``.  Overrides :data:`DEFAULT_X0`.
    regularization : dict, optional
        ``{param_name: (prior_mean, prior_sigma)}`` for Gaussian priors.
    wavelength_mask : np.ndarray of bool, optional
        Wavelength mask for spectral mode (True = include).
    method : str
        ``"L-BFGS-B"`` (default), ``"Nelder-Mead"``,
        ``"differential_evolution"``, or ``"mcmc"``.
    mcmc_walkers : int
        Number of MCMC walkers (only for ``method="mcmc"``).
    mcmc_steps : int
        Number of MCMC steps.
    mcmc_burn : int
        Number of burn-in steps to discard.
    fixed_params : dict, optional
        ``{param_name: value}`` for parameters that are known and should
        not be optimised.  The parameter must exist in the emulator's
        input space.

    Returns
    -------
    RetrievalResult
    """
    observed = np.asarray(observed, dtype=float)

    # --- Validate inputs ---
    if emulator is None and forward_fn is None:
        raise ValueError("Provide either `emulator` or `forward_fn`.")
    if platform is not None and observed_band_names is None:
        raise ValueError("`observed_band_names` is required when `platform` is set.")

    # --- Build forward function from emulator ---
    if emulator is not None and forward_fn is None:
        forward_fn = _make_emulator_fn(emulator, parameters, fixed_params)
        if bounds is None:
            bounds = {}
        # Fill bounds from emulator where not overridden
        emu_bounds = emulator.bounds
        for p in parameters:
            if p not in bounds:
                if p in emu_bounds:
                    bounds[p] = emu_bounds[p]
                elif p in DEFAULT_BOUNDS:
                    bounds[p] = DEFAULT_BOUNDS[p]

    # --- Resolve bounds and x0 ---
    if bounds is None:
        bounds = {}
    active_bounds = []
    for p in parameters:
        if p in bounds:
            active_bounds.append(bounds[p])
        elif p in DEFAULT_BOUNDS:
            active_bounds.append(DEFAULT_BOUNDS[p])
        else:
            raise ValueError(
                "No bounds for parameter {!r}. "
                "Provide via `bounds` argument.".format(p)
            )

    x0_dict = dict(DEFAULT_X0)
    if x0 is not None:
        x0_dict.update(x0)
    x0_vec = np.array([float(x0_dict.get(p, np.mean(active_bounds[i])))
                        for i, p in enumerate(parameters)])

    # --- Build cost function ---
    flx_slr = getattr(emulator, "flx_slr", None) if emulator else None

    if platform is not None:
        # Band mode
        if flx_slr is None:
            raise ValueError(
                "Band mode requires `flx_slr`.  Use an emulator or "
                "provide a forward_fn that doesn't need band convolution."
            )

        def cost_fn(params):
            return band_cost(
                params, parameters, observed, observed_band_names,
                forward_fn, flx_slr, platform,
                obs_uncertainty=obs_uncertainty,
                regularization=regularization,
            )
    else:
        # Spectral mode
        def cost_fn(params):
            return spectral_cost(
                params, parameters, observed, forward_fn,
                obs_uncertainty=obs_uncertainty,
                wavelength_mask=wavelength_mask,
                regularization=regularization,
            )

    # --- Dispatch to optimiser ---
    if method == "mcmc":
        return _run_mcmc(
            cost_fn, parameters, active_bounds, x0_vec,
            forward_fn, observed, method,
            mcmc_walkers, mcmc_steps, mcmc_burn,
        )
    elif method == "differential_evolution":
        return _run_differential_evolution(
            cost_fn, parameters, active_bounds, forward_fn, observed, method,
        )
    else:
        return _run_scipy_minimize(
            cost_fn, parameters, active_bounds, x0_vec,
            forward_fn, observed, method,
        )


def _make_emulator_fn(emulator, parameters, fixed_params):
    """Build a forward function that fills in fixed params for the emulator."""
    all_params = emulator.param_names
    fixed = dict(fixed_params) if fixed_params else {}

    def fn(**active_params):
        full = dict(fixed)
        full.update(active_params)
        return emulator.predict(**full)

    return fn


def _run_scipy_minimize(cost_fn, parameters, active_bounds, x0_vec,
                        forward_fn, observed, method):
    """Run scipy.optimize.minimize and return RetrievalResult."""
    options = {"maxiter": 2000}
    if method == "L-BFGS-B":
        options["ftol"] = 1e-12
    elif method == "Nelder-Mead":
        options["fatol"] = 1e-12
        options["xatol"] = 1e-10

    result = minimize(
        cost_fn,
        x0_vec,
        method=method,
        bounds=active_bounds if method != "Nelder-Mead" else None,
        options=options,
    )

    best_fit = dict(zip(parameters, result.x))
    predicted = forward_fn(**best_fit)

    # Hessian uncertainty
    uncertainty = _hessian_uncertainty(
        cost_fn, result.x, parameters, active_bounds
    )

    return RetrievalResult(
        best_fit=best_fit,
        cost=float(result.fun),
        uncertainty=uncertainty,
        predicted_albedo=predicted,
        observed=observed,
        converged=bool(result.success),
        method=method,
        n_function_evals=int(result.nfev),
    )


def _run_differential_evolution(cost_fn, parameters, active_bounds,
                                forward_fn, observed, method):
    """Run differential_evolution and return RetrievalResult."""
    result = differential_evolution(
        cost_fn,
        bounds=active_bounds,
        maxiter=1000,
        tol=1e-10,
        seed=42,
    )

    best_fit = dict(zip(parameters, result.x))
    predicted = forward_fn(**best_fit)

    uncertainty = _hessian_uncertainty(
        cost_fn, result.x, parameters, active_bounds
    )

    return RetrievalResult(
        best_fit=best_fit,
        cost=float(result.fun),
        uncertainty=uncertainty,
        predicted_albedo=predicted,
        observed=observed,
        converged=bool(result.success),
        method=method,
        n_function_evals=int(result.nfev),
    )


def _run_mcmc(cost_fn, parameters, active_bounds, x0_vec,
              forward_fn, observed, method,
              n_walkers, n_steps, n_burn):
    """Run emcee MCMC sampler and return RetrievalResult."""
    try:
        import emcee
    except ImportError:
        raise ImportError(
            "emcee is required for MCMC.  Install with:  pip install emcee>=3"
        )

    n_params = len(parameters)
    lo = np.array([b[0] for b in active_bounds])
    hi = np.array([b[1] for b in active_bounds])

    def log_prob(params):
        # Uniform prior: -inf if outside bounds
        if np.any(params < lo) or np.any(params > hi):
            return -np.inf
        return -0.5 * cost_fn(params)

    # Initialise walkers as Gaussian ball around x0
    spread = (hi - lo) * 0.01
    p0 = x0_vec + spread * np.random.randn(n_walkers, n_params)
    # Clip to bounds
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)

    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_prob)
    sampler.run_mcmc(p0, n_steps + n_burn, progress=False)

    chains = sampler.get_chain(discard=n_burn)  # (n_steps, n_walkers, n_params)
    flat = chains.reshape(-1, n_params)

    best_fit = dict(zip(parameters, np.median(flat, axis=0)))
    uncertainty = dict(zip(parameters, np.std(flat, axis=0)))
    predicted = forward_fn(**best_fit)

    # Acceptance fraction
    acc = float(np.mean(sampler.acceptance_fraction))

    # Autocorrelation time (may fail if chains are too short)
    try:
        autocorr = sampler.get_autocorr_time(quiet=True)
    except Exception:
        autocorr = np.full(n_params, np.nan)

    cost_at_median = cost_fn(np.array([best_fit[p] for p in parameters]))

    return RetrievalResult(
        best_fit=best_fit,
        cost=float(cost_at_median),
        uncertainty=uncertainty,
        predicted_albedo=predicted,
        observed=observed,
        converged=acc > 0.1,  # crude convergence check
        method=method,
        n_function_evals=n_walkers * (n_steps + n_burn),
        chains=chains,
        acceptance_fraction=acc,
        autocorr_time=autocorr,
    )


def _hessian_uncertainty(cost_fn, x_opt, parameters, active_bounds):
    """Compute parameter uncertainties from finite-difference Hessian.

    Returns ``{param_name: 1_sigma}`` from the diagonal of the
    inverse Hessian at the optimum.  Parameters whose Hessian row
    is singular get ``inf`` uncertainty.
    """
    n = len(x_opt)
    # Step sizes: 1e-4 of parameter range
    steps = np.array([(b[1] - b[0]) * 1e-4 for b in active_bounds])
    steps = np.maximum(steps, 1e-10)

    hessian = np.zeros((n, n))
    f0 = cost_fn(x_opt)

    for i in range(n):
        for j in range(i, n):
            x_pp = x_opt.copy()
            x_pm = x_opt.copy()
            x_mp = x_opt.copy()
            x_mm = x_opt.copy()

            x_pp[i] += steps[i]
            x_pp[j] += steps[j]

            x_pm[i] += steps[i]
            x_pm[j] -= steps[j]

            x_mp[i] -= steps[i]
            x_mp[j] += steps[j]

            x_mm[i] -= steps[i]
            x_mm[j] -= steps[j]

            hessian[i, j] = (
                cost_fn(x_pp) - cost_fn(x_pm) - cost_fn(x_mp) + cost_fn(x_mm)
            ) / (4.0 * steps[i] * steps[j])
            hessian[j, i] = hessian[i, j]

    # Invert to get covariance; fall back to inf for singular Hessian
    try:
        cov = np.linalg.inv(hessian)
        sigmas = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        sigmas = np.full(n, float("inf"))

    return dict(zip(parameters, sigmas))
