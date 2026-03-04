"""Neural network emulator for the BioSNICAR forward model.

Build phase uses scikit-learn for MLP training and PCA fitting.
Predict phase uses pure numpy — no sklearn dependency at inference.

The builder is general-purpose: any ``run_model()`` keyword can be a
free parameter or a fixed override.  The default configuration targets
solid glacier ice, but custom emulators can be built for snow, different
impurity sets, or any other model configuration.
"""

import json
import warnings
from collections import OrderedDict
from datetime import datetime

import numpy as np


def _latin_hypercube(n_samples, n_dims, seed=None):
    """Generate Latin hypercube samples in [0, 1]^d.

    Each dimension is stratified into *n_samples* equal intervals with
    exactly one sample per stratum, then columns are independently
    shuffled.  This gives uniform marginals and good space-filling
    without requiring ``scipy.stats.qmc`` (added in scipy 1.7).

    Parameters
    ----------
    n_samples : int
        Number of sample points.
    n_dims : int
        Number of dimensions.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_samples, n_dims)
        Samples in [0, 1]^d.
    """
    rng = np.random.RandomState(seed)
    samples = np.empty((n_samples, n_dims))
    for j in range(n_dims):
        # One sample per stratum, uniformly within the stratum
        strata = np.arange(n_samples)
        rng.shuffle(strata)
        samples[:, j] = (strata + rng.uniform(size=n_samples)) / n_samples
    return samples


# Binary parameters are snapped to {0, 1} during LHS.
_BINARY_PARAMS = {"direct"}

# Parameters that require integer rounding before passing to run_model.
# The optical-property lookup tables use integer keys (e.g. rds on a
# 5/10/500 grid), so continuous LHS values must be snapped.
_INTEGER_PARAMS = {"rds"}


def _snap_rds(value):
    """Snap a continuous rds value to the nearest lookup-table radius.

    The bubbly-ice LUT grid is: step 5 for [10, 100), step 10 for
    [100, 5000], step 500 for (5000, 25000].
    """
    v = float(value)
    if v < 100:
        return int(round(v / 5) * 5)
    elif v <= 5000:
        return int(round(v / 10) * 10)
    else:
        return int(round(v / 500) * 500)


class Emulator:
    """Fast neural-network surrogate for the BioSNICAR forward model.

    Build with :meth:`build` (requires scikit-learn), then predict with
    :meth:`predict` (pure numpy, ~microseconds per call).  Save/load
    via :meth:`save` / :meth:`load` using a portable ``.npz`` format
    that does not require scikit-learn.

    Examples
    --------
    >>> emu = Emulator.build(
    ...     params={"rds": (100, 5000), "impurity_0_conc": (0, 50000)},
    ...     n_samples=500, solzen=50, progress=False,
    ... )
    >>> albedo = emu.predict(rds=800, impurity_0_conc=5000)
    >>> emu.save("my_emulator.npz")
    >>> emu2 = Emulator.load("my_emulator.npz")
    """

    def __init__(self):
        # Populated by build() or load()
        self._weights = []       # list of 2-D weight matrices
        self._biases = []        # list of 1-D bias vectors
        self._pca_components = None  # (n_pca, 480)
        self._pca_mean = None        # (480,)
        self._input_min = None       # (n_params,)
        self._input_max = None       # (n_params,)
        self._flx_slr = None         # (480,)
        self._param_names = []       # ordered list of param names
        self._bounds = OrderedDict() # {name: (lo, hi)}
        self._n_pca_components = 0
        self._training_score = None
        self._metadata = {}

    # ── Properties ────────────────────────────────────────────────────

    @property
    def param_names(self):
        """Ordered list of parameter names."""
        return list(self._param_names)

    @property
    def bounds(self):
        """Parameter bounds ``{name: (min, max)}``."""
        return dict(self._bounds)

    @property
    def n_pca_components(self):
        """Number of PCA components retained."""
        return self._n_pca_components

    @property
    def training_score(self):
        """R² score on held-out validation set (from build)."""
        return self._training_score

    @property
    def flx_slr(self):
        """480-element solar flux spectrum stored at build time."""
        return self._flx_slr

    # ── Build ─────────────────────────────────────────────────────────

    @classmethod
    def build(cls, params, n_samples=10000, solver="adding-doubling",
              input_file="default", progress=True, seed=42,
              **fixed_overrides):
        """Build an emulator by training an MLP on forward-model outputs.

        Parameters
        ----------
        params : dict
            ``{param_name: (min_value, max_value)}`` for each free
            parameter.  Any keyword accepted by ``run_model()`` can be
            used (e.g. ``rds``, ``rho``, ``solzen``, ``direct``,
            ``impurity_0_conc``, etc.).
        n_samples : int
            Number of Latin hypercube training samples.  More samples
            improve accuracy but increase build time (~50 ms per sample).
        solver : str
            ``"adding-doubling"`` (default) or ``"toon"``.
        input_file : str
            Path to YAML config, or ``"default"`` for the built-in
            ``inputs.yaml``.
        progress : bool
            Show a tqdm progress bar during forward-model runs.
        seed : int
            Random seed for Latin hypercube sampling and MLP training.
        **fixed_overrides
            Fixed parameters passed to every ``run_model()`` call.
            For glacier ice, ``layer_type=1`` is typical.

        Returns
        -------
        Emulator
            Trained emulator ready for :meth:`predict` calls.
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.neural_network import MLPRegressor
        except ImportError:
            raise ImportError(
                "scikit-learn is required to build an emulator.  "
                "Install it with:  pip install scikit-learn>=1.0\n"
                "Once built, the emulator can be loaded and used "
                "without scikit-learn."
            )

        from biosnicar.drivers.run_model import run_model

        param_names = list(params.keys())
        bounds_ordered = OrderedDict(
            (name, (float(lo), float(hi))) for name, (lo, hi) in params.items()
        )
        n_dims = len(param_names)

        # --- 1. Latin hypercube sampling ---
        lhs_unit = _latin_hypercube(n_samples, n_dims, seed=seed)

        # Scale to parameter bounds
        lo_arr = np.array([bounds_ordered[n][0] for n in param_names])
        hi_arr = np.array([bounds_ordered[n][1] for n in param_names])
        lhs_scaled = lo_arr + lhs_unit * (hi_arr - lo_arr)

        # Snap binary parameters to {0, 1}
        for j, name in enumerate(param_names):
            if name in _BINARY_PARAMS:
                lhs_scaled[:, j] = np.round(lhs_scaled[:, j]).astype(float)

        # --- 2. Run forward model for each sample ---
        albedos = []
        flx_slr_ref = None

        iterator = range(n_samples)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Building emulator")
            except ImportError:
                pass

        for i in iterator:
            overrides = dict(fixed_overrides)
            for j, name in enumerate(param_names):
                val = lhs_scaled[i, j]
                # Snap to valid lookup-table values
                if name == "rds":
                    val = _snap_rds(val)
                elif name in _BINARY_PARAMS:
                    val = int(val)
                elif name in _INTEGER_PARAMS:
                    val = int(round(val))
                overrides[name] = val

            outputs = run_model(input_file=input_file, solver=solver,
                                **overrides)
            albedos.append(np.array(outputs.albedo, dtype=np.float64))
            if flx_slr_ref is None:
                flx_slr_ref = np.array(outputs.flx_slr, dtype=np.float64)

        albedo_matrix = np.array(albedos)  # (n_samples, 480)

        # --- 3. PCA compression ---
        pca = PCA(n_components=0.999)  # retain 99.9% variance
        pca_coeffs = pca.fit_transform(albedo_matrix)
        n_pca = pca.n_components_

        # --- 4. Min-max scale inputs to [0, 1] ---
        input_min = lo_arr.copy()
        input_max = hi_arr.copy()
        X_scaled = (lhs_scaled - input_min) / (input_max - input_min + 1e-30)

        # --- 5. Train MLP ---
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 128, 64),
            activation="relu",
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
        mlp.fit(X_scaled, pca_coeffs)

        # Validation score (R² on held-out validation set)
        r2_score = mlp.score(X_scaled, pca_coeffs)

        # --- 6. Extract weights and assemble emulator ---
        emu = cls()
        emu._param_names = list(param_names)
        emu._bounds = bounds_ordered
        emu._input_min = input_min
        emu._input_max = input_max
        emu._pca_components = pca.components_.copy()  # (n_pca, 480)
        emu._pca_mean = pca.mean_.copy()              # (480,)
        emu._n_pca_components = n_pca
        emu._flx_slr = flx_slr_ref
        emu._training_score = float(r2_score)

        # Extract MLP weights/biases as numpy arrays
        emu._weights = [w.copy() for w in mlp.coefs_]
        emu._biases = [b.copy() for b in mlp.intercepts_]

        emu._metadata = {
            "param_names": list(param_names),
            "bounds": {n: list(v) for n, v in bounds_ordered.items()},
            "n_samples": int(n_samples),
            "n_pca_components": int(n_pca),
            "training_r2": float(r2_score),
            "build_timestamp": datetime.utcnow().isoformat(),
            "fixed_overrides": {k: _jsonable(v) for k, v in fixed_overrides.items()},
            "solver": solver,
        }

        return emu

    # ── Predict ───────────────────────────────────────────────────────

    def predict(self, **params):
        """Predict 480-band spectral albedo (~microseconds).

        Parameters
        ----------
        **params
            Keyword arguments for each emulator parameter
            (e.g. ``rds=800, impurity_0_conc=5000``).

        Returns
        -------
        np.ndarray of shape (480,)
            Predicted spectral albedo, clipped to [0, 1].
        """
        x = self._params_to_scaled(params)
        return self._forward_pass(x).ravel()

    def predict_batch(self, points):
        """Predict for N parameter combinations.

        Parameters
        ----------
        points : np.ndarray of shape (N, n_params)
            Each row is one parameter set in the order of
            :attr:`param_names`.

        Returns
        -------
        np.ndarray of shape (N, 480)
            Predicted spectral albedo per row.
        """
        points = np.atleast_2d(points)
        # Scale
        x = (points - self._input_min) / (self._input_max - self._input_min + 1e-30)
        x = np.clip(x, 0.0, 1.0)
        return self._forward_pass(x)

    def _params_to_scaled(self, params_dict):
        """Convert keyword params to a scaled (0-1) 1-D array."""
        missing = set(self._param_names) - set(params_dict.keys())
        if missing:
            raise ValueError(f"Missing parameters: {missing}")
        x = np.array([float(params_dict[n]) for n in self._param_names])
        # Warn if out of bounds
        for i, name in enumerate(self._param_names):
            lo, hi = self._bounds[name]
            if x[i] < lo or x[i] > hi:
                warnings.warn(
                    f"Parameter {name}={x[i]} is outside training bounds "
                    f"[{lo}, {hi}].  Predictions may be unreliable.",
                    stacklevel=3,
                )
        x = (x - self._input_min) / (self._input_max - self._input_min + 1e-30)
        x = np.clip(x, 0.0, 1.0)
        return x.reshape(1, -1)

    def _forward_pass(self, x):
        """Pure-numpy MLP forward pass + PCA reconstruction.

        Parameters
        ----------
        x : np.ndarray of shape (N, n_params)
            Scaled inputs in [0, 1].

        Returns
        -------
        np.ndarray of shape (N, 480)
        """
        for i, (W, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ W + b
            if i < len(self._weights) - 1:
                x = np.maximum(x, 0)  # ReLU on all but last layer
        # x is now PCA coefficients of shape (N, n_pca)
        albedo = x @ self._pca_components + self._pca_mean
        return np.clip(albedo, 0.0, 1.0)

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, path):
        """Save emulator to a ``.npz`` file.

        The file contains only numpy arrays and a JSON metadata string
        — no pickle, no sklearn objects.  This ensures portability
        across Python and sklearn versions.

        Parameters
        ----------
        path : str or Path
            Output file path (should end in ``.npz``).
        """
        arrays = {}
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            arrays["weights_{}".format(i)] = w
            arrays["biases_{}".format(i)] = b
        arrays["pca_components"] = self._pca_components
        arrays["pca_mean"] = self._pca_mean
        arrays["input_min"] = self._input_min
        arrays["input_max"] = self._input_max
        arrays["flx_slr"] = self._flx_slr
        arrays["metadata"] = np.array(json.dumps(self._metadata))
        np.savez_compressed(str(path), **arrays)

    @classmethod
    def load(cls, path):
        """Load emulator from a ``.npz`` file.  No sklearn required.

        Parameters
        ----------
        path : str or Path
            Path to a ``.npz`` file produced by :meth:`save`.

        Returns
        -------
        Emulator
        """
        data = np.load(str(path), allow_pickle=False)
        meta = json.loads(str(data["metadata"]))

        emu = cls()
        emu._param_names = meta["param_names"]
        emu._bounds = OrderedDict(
            (n, tuple(meta["bounds"][n])) for n in meta["param_names"]
        )
        emu._n_pca_components = meta["n_pca_components"]
        emu._training_score = meta.get("training_r2")
        emu._metadata = meta

        emu._pca_components = data["pca_components"]
        emu._pca_mean = data["pca_mean"]
        emu._input_min = data["input_min"]
        emu._input_max = data["input_max"]
        emu._flx_slr = data["flx_slr"]

        # Load weight/bias pairs
        i = 0
        while "weights_{}".format(i) in data:
            emu._weights.append(data["weights_{}".format(i)])
            emu._biases.append(data["biases_{}".format(i)])
            i += 1

        return emu

    def __repr__(self):
        return (
            "Emulator(params={}, n_pca={}, training_r2={})".format(
                self._param_names, self._n_pca_components,
                self._training_score,
            )
        )


def _jsonable(v):
    """Convert a value to a JSON-serialisable type."""
    if isinstance(v, (np.integer, np.int_)):
        return int(v)
    if isinstance(v, (np.floating, np.float_)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
