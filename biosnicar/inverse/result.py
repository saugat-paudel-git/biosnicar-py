"""RetrievalResult dataclass for inversion outputs.

Holds the best-fit parameters, uncertainties, predicted spectrum, convergence
diagnostics, and (optionally) MCMC chains from a ``retrieve()`` call.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class RetrievalResult:
    """Result container returned by :func:`~biosnicar.inverse.optimize.retrieve`.

    Attributes
    ----------
    best_fit : dict
        ``{param_name: best_value}`` at the cost-function minimum.
    cost : float
        Final cost-function value (chi-squared).
    uncertainty : dict
        ``{param_name: 1_sigma}`` from Hessian or MCMC posterior.
    predicted_albedo : np.ndarray
        480-band spectral albedo at the best-fit point.
    observed : np.ndarray
        Input observations (480-band array or N-band satellite array).
    converged : bool
        Whether the optimiser reported convergence.
    method : str
        Optimisation method used (e.g. ``"L-BFGS-B"``, ``"mcmc"``).
    n_function_evals : int
        Number of forward-model (or emulator) evaluations.
    chains : np.ndarray or None
        MCMC chains of shape ``(n_steps, n_walkers, n_params)``.
        Only populated when ``method="mcmc"``.
    acceptance_fraction : float or None
        Mean MCMC acceptance fraction.  Only for ``method="mcmc"``.
    autocorr_time : np.ndarray or None
        Integrated autocorrelation time per parameter.  Only for ``method="mcmc"``.
    """

    best_fit: Dict[str, float]
    cost: float
    uncertainty: Dict[str, float]
    predicted_albedo: np.ndarray
    observed: np.ndarray
    converged: bool
    method: str
    n_function_evals: int

    # MCMC-specific fields (None unless method="mcmc")
    chains: Optional[np.ndarray] = None
    acceptance_fraction: Optional[float] = None
    autocorr_time: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [f"RetrievalResult (method={self.method}, converged={self.converged})"]
        lines.append(f"  Cost: {self.cost:.4f}  |  Function evaluations: {self.n_function_evals}")
        lines.append("  Best-fit parameters:")
        for name, val in self.best_fit.items():
            unc = self.uncertainty.get(name, float("nan"))
            lines.append(f"    {name:25s} = {val:12.4f}  ±  {unc:.4f}")
        if self.chains is not None:
            n_steps, n_walkers, n_params = self.chains.shape
            lines.append(
                f"  MCMC: {n_steps} steps × {n_walkers} walkers, "
                f"acceptance = {self.acceptance_fraction:.3f}"
            )
        return "\n".join(lines)
