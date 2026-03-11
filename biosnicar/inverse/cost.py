"""Cost functions for spectral and satellite-band retrieval.

Provides chi-squared cost functions used by :func:`~biosnicar.inverse.optimize.retrieve`
to compare emulator (or forward-model) predictions against observations.

Two modes are supported:

* **Spectral mode** — compare 480-band predicted albedo to 480-band observed
  albedo.  Used with field spectrometer data or full-spectrum retrievals.

* **Band mode** — predict 480-band albedo, convolve to satellite bands via
  ``biosnicar.bands.to_platform``, compare to observed satellite band values.
  Used for native Sentinel-2, Landsat 8, MODIS, etc. inversions.
"""

import numpy as np


def spectral_cost(params, param_names, observed, forward_fn,
                  obs_uncertainty=None, wavelength_mask=None,
                  regularization=None):
    """Chi-squared cost over 480-band spectral albedo.

    .. math::

        J = \\sum_i \\frac{(\\hat{\\alpha}_i - \\alpha_i^{\\text{obs}})^2}{\\sigma_i^2}
            + \\sum_k \\frac{(\\theta_k - \\mu_k)^2}{\\sigma_{\\text{prior},k}^2}

    Parameters
    ----------
    params : array-like, shape (n_params,)
        Current parameter values (scipy.optimize format).
    param_names : list of str
        Ordered parameter names corresponding to *params*.
    observed : np.ndarray, shape (480,)
        Observed spectral albedo.
    forward_fn : callable
        ``forward_fn(**param_dict) -> np.ndarray (480,)``
    obs_uncertainty : np.ndarray, shape (480,), optional
        Per-wavelength 1-sigma measurement uncertainty.
        Defaults to 1.0 (unweighted least squares).
    wavelength_mask : np.ndarray of bool, shape (480,), optional
        ``True`` for wavelengths to include; ``False`` to exclude.
    regularization : dict, optional
        ``{param_name: (prior_mean, prior_sigma)}`` for Gaussian
        regularization terms.

    Returns
    -------
    float
        Scalar cost value (non-negative).
    """
    param_dict = dict(zip(param_names, params))
    predicted = forward_fn(**param_dict)

    residual = predicted - observed

    if wavelength_mask is not None:
        residual = residual[wavelength_mask]
        if obs_uncertainty is not None:
            obs_uncertainty = obs_uncertainty[wavelength_mask]

    if obs_uncertainty is not None:
        residual = residual / obs_uncertainty

    cost = float(np.sum(residual ** 2))

    if regularization:
        for name, (prior_mean, prior_sigma) in regularization.items():
            if name in param_dict:
                cost += ((param_dict[name] - prior_mean) / prior_sigma) ** 2

    return cost


def band_cost(params, param_names, observed, observed_band_names,
              forward_fn, flx_slr, platform,
              obs_uncertainty=None, regularization=None):
    """Chi-squared cost over satellite band albedo values.

    Predicts 480-band albedo, convolves to platform bands via
    :func:`biosnicar.bands.to_platform`, then computes chi-squared
    against observed band values.

    Parameters
    ----------
    params : array-like, shape (n_params,)
        Current parameter values.
    param_names : list of str
        Ordered parameter names.
    observed : np.ndarray, shape (n_bands,)
        Observed satellite band albedo values.
    observed_band_names : list of str
        Band names corresponding to elements of *observed*
        (e.g. ``["B3", "B4", "B11"]``).
    forward_fn : callable
        ``forward_fn(**param_dict) -> np.ndarray (480,)``
    flx_slr : np.ndarray, shape (480,)
        Spectral solar flux for band convolution.
    platform : str
        Platform key (e.g. ``"sentinel2"``, ``"modis"``).
    obs_uncertainty : np.ndarray, shape (n_bands,), optional
        Per-band 1-sigma measurement uncertainty.
        Defaults to 1.0 (unweighted).
    regularization : dict, optional
        ``{param_name: (prior_mean, prior_sigma)}``.

    Returns
    -------
    float
        Scalar cost value (non-negative).
    """
    from biosnicar.bands import to_platform

    param_dict = dict(zip(param_names, params))
    predicted_albedo = forward_fn(**param_dict)

    band_result = to_platform(predicted_albedo, platform, flx_slr=flx_slr)

    # Extract predicted band values in the order of observed_band_names
    predicted_bands = np.array(
        [getattr(band_result, name) for name in observed_band_names]
    )

    residual = predicted_bands - observed

    if obs_uncertainty is not None:
        residual = residual / obs_uncertainty

    cost = float(np.sum(residual ** 2))

    if regularization:
        for name, (prior_mean, prior_sigma) in regularization.items():
            if name in param_dict:
                cost += ((param_dict[name] - prior_mean) / prior_sigma) ** 2

    return cost
