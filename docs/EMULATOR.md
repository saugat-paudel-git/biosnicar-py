# Emulator

The BioSNICAR emulator is a neural-network surrogate that replaces the forward model for fast spectral albedo prediction. A single forward model call takes ~50 ms; the emulator predicts in ~microseconds, making optimisation and MCMC practical.

## Quick Start

```python
from biosnicar.emulator import Emulator
from biosnicar import run_emulator

# Load the pre-built default emulator (ships with the repo)
emu = Emulator.load("data/emulators/glacier_ice_7_param_default.npz") #glacier_ice_7_param_default

# Predict spectral albedo
albedo = emu.predict(rds=1000, rho=600, black_carbon=5000, glacier_algae=50000)

# Or get a full Outputs object (BBA, BBAVIS, BBANIR, .to_platform())
outputs = run_emulator(emu, rds=1000, rho=600, black_carbon=5000, glacier_algae=50000)
print(outputs.BBA)
outputs.to_platform("sentinel2")
```

### Build a custom emulator

```python
emu = Emulator.build(
    params={
        "rds": (100, 5000),
        "rho": (100, 917),
        "black_carbon": (0, 100000),
        "glacier_algae": (0, 500000),
    },
    n_samples=5000,
    layer_type=1,       # fixed: solid ice
    solzen=50,          # fixed: solar zenith angle
    direct=1,           # fixed: clear sky
)
emu.save("my_emulator.npz")
```

### Save and load

```python
emu.save("glacier_ice.npz")       # ~100-200 KB, no pickle, no sklearn
emu2 = Emulator.load("glacier_ice.npz")  # pure numpy, no sklearn required
```

## API Reference

### `Emulator.build(params, n_samples=10000, solver="adding-doubling", input_file="default", progress=True, seed=42, **fixed_overrides)`

Train an emulator on Latin hypercube samples of the forward model.

| Parameter           | Type   | Default             | Description                                                                    |
| ------------------- | ------ | ------------------- | ------------------------------------------------------------------------------ |
| `params`            | dict   | required            | `{name: (min, max)}` for each free parameter. Any `run_model()` keyword works. |
| `n_samples`         | int    | 10000               | Number of LHS training samples. More = better accuracy, longer build.          |
| `solver`            | str    | `"adding-doubling"` | RT solver: `"adding-doubling"` or `"toon"`.                                    |
| `input_file`        | str    | `"default"`         | YAML config path. `"default"` uses bundled `inputs.yaml`.                      |
| `progress`          | bool   | True                | Show tqdm progress bar.                                                        |
| `seed`              | int    | 42                  | Random seed for LHS and MLP training.                                          |
| `**fixed_overrides` | kwargs | —                   | Fixed parameters for every forward run (e.g. `layer_type=1`).                  |

**Returns:** `Emulator` instance.

**Requires:** `scikit-learn>=1.0` (build-time only).

### `Emulator.predict(**params)`

Predict 480-band spectral albedo. Pure numpy, ~microseconds.

| Parameter  | Type   | Description                                           |
| ---------- | ------ | ----------------------------------------------------- |
| `**params` | kwargs | One keyword per emulator parameter (e.g. `rds=1000`). |

**Returns:** `np.ndarray` of shape `(480,)`, clipped to [0, 1].

### `Emulator.predict_batch(points)`

Predict for N parameter combinations.

| Parameter | Type                  | Description                                     |
| --------- | --------------------- | ----------------------------------------------- |
| `points`  | ndarray (N, n_params) | Rows are parameter sets in `param_names` order. |

**Returns:** `np.ndarray` of shape `(N, 480)`.

### `Emulator.save(path)` / `Emulator.load(path)`

Save/load emulator to/from `.npz`. No sklearn required for loading.

### Properties

| Property           | Type           | Description                          |
| ------------------ | -------------- | ------------------------------------ |
| `param_names`      | list[str]      | Ordered parameter names.             |
| `bounds`           | dict           | `{name: (min, max)}` from training.  |
| `n_pca_components` | int            | Number of PCA components retained.   |
| `training_score`   | float          | R² on training data.                 |
| `flx_slr`          | ndarray (480,) | Solar flux spectrum from build time. |

### `run_emulator(emulator, **params)`

Wrapper that returns an `Outputs` object (same interface as `run_model()`), with `.albedo`, `.BBA`, `.BBAVIS`, `.BBANIR`, `.flx_slr`, and `.to_platform()`. Fields only available from the full RT solver (`heat_rt`, `absorbed_flux_per_layer`) are `None`.

## Design Rationale

### Why an emulator?

The forward model takes ~50 ms per evaluation. This limits what you can do:

| Approach                              | Evaluations | Wall time     |
| ------------------------------------- | ----------- | ------------- |
| Direct optimisation (scipy)           | 100-500     | 5-25 s        |
| Global optimisation (diff. evolution) | 500-5000    | 25 s - 4 min  |
| MCMC (32 walkers x 5000 steps)        | 160,000     | ~2.2 hours    |
| **Emulator** optimisation             | 100-500     | **< 100 ms**  |
| **Emulator** MCMC                     | 160,000     | **~1 minute** |

The emulator makes both fast optimisation and full Bayesian uncertainty quantification practical.

### Why MLP, not grid interpolation?

The curse of dimensionality makes Cartesian grids scale as O(N^d):

| Parameters | Grid (20 pts/axis)       | MLP (LHS)             |
| ---------- | ------------------------ | --------------------- |
| 2          | 400 runs / ~10 MB        | 5,000 runs / ~100 KB  |
| 3          | 8,000 runs / ~35 MB      | 5,000 runs / ~100 KB  |
| 5          | 3,200,000 runs / ~1.2 GB | 10,000 runs / ~150 KB |
| 6          | 64,000,000 / infeasible  | 10,000 runs / ~150 KB |

A realistic glacier ice retrieval needs 4-7 parameters. Grid interpolation is infeasible beyond 3; the MLP scales linearly.

### Alternatives considered

| Approach                    | Why not                                                                                   |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| **Gaussian Process**        | O(n³) scaling. Impractical for 5000+ training points. Prediction ~ms (too slow for MCMC). |
| **RBF interpolation**       | Dense O(n²) memory, O(n³) solve. Stores all training points.                              |
| **Polynomial regression**   | Cannot capture nonlinear spectral features (absorption bands).                            |
| **Random forest / XGBoost** | Step-function approximation. Not differentiable — breaks gradient-based optimisers.       |

**MLP wins** on: O(N) scaling, microsecond inference, smooth differentiable output, compact storage (~100 KB), pure-numpy inference.

### Why PCA compression?

Raw spectral output is 480 bands. PCA typically compresses to ~10 components while retaining >99.9% of variance. Benefits:

- **Regularises training** — the MLP learns smooth spectral structure, not noise
- **Reduces network size** — 10 outputs instead of 480
- **Compact storage** — ~100 KB instead of multi-MB
- **Often improves accuracy** — captures dominant spectral modes

### Why scikit-learn for training, numpy for inference?

- **Build time** uses `sklearn.neural_network.MLPRegressor` and `sklearn.decomposition.PCA`. Writing a correct MLP training loop from scratch would be hundreds of error-prone lines. sklearn handles backprop, Adam, early stopping, and train/validation splits.
- **Inference** extracts weights as numpy arrays. The forward pass is ~4 matrix multiplications. No sklearn import needed.

This means users who load a pre-built `.npz` file never need scikit-learn installed. Deployment environments stay lightweight.

Why not PyTorch/TensorFlow? They're ~500 MB+ dependencies, overkill for a 3-layer MLP.

### Why `.npz`, not pickle?

- **Version-independent** — numpy arrays don't break across Python/sklearn versions
- **No arbitrary code execution** — safe to share (unlike pickle)
- **Human-inspectable** — can be loaded and examined with `np.load()`
- **Compact** — compressed numpy arrays, ~100-200 KB

### Why 128-128-64 ReLU architecture?

Ice albedo is a smooth function of physical parameters. This modest network is an empirical sweet spot:
- **Deeper networks** (256-256-128) show marginal accuracy gain but triple storage and inference time
- **Shallower networks** (64-32) underfit spectral features near absorption bands
- **ReLU** is sufficient for smooth physical functions and enables fast inference (no exp/tanh)

## Training Details

### Latin hypercube sampling

LHS fills parameter space uniformly with far fewer samples than a Cartesian grid. Each parameter's marginal distribution is stratified into N equal bins with one sample per bin. This provides better coverage than random sampling.

### Parameter snapping

- `rds` values are snapped to the nearest lookup-table entry (step 5 for rds < 100, step 10 for 100-5000, step 500 for >5000)
- `direct` is snapped to {0, 1} — the MLP sees equal numbers of clear/cloudy samples

### MLP training

```python
MLPRegressor(
    hidden_layer_sizes=(128, 128, 64),
    activation="relu",
    max_iter=2000,
    early_stopping=True,        # stops when validation loss plateaus
    validation_fraction=0.1,    # 10% held out
    random_state=42,
)
```

### Build time

| Samples | Build time | Storage | Suitable for               |
| ------- | ---------- | ------- | -------------------------- |
| 5,000   | ~4 min     | ~100 KB | 2-4 parameter emulators    |
| 10,000  | ~8 min     | ~150 KB | 4-5 parameter emulators    |
| 15,000  | ~12 min    | ~180 KB | 6+ parameter emulators     |
| 20,000  | ~17 min    | ~200 KB | High-accuracy applications |

Build time is dominated by forward model runs (~50 ms each). MLP training adds <10 seconds.

## `.npz` File Format

| Key                       | Shape           | Description                                      |
| ------------------------- | --------------- | ------------------------------------------------ |
| `weights_0`               | (n_params, 128) | First layer weights                              |
| `weights_1`               | (128, 128)      | Second layer weights                             |
| `weights_2`               | (128, 64)       | Third layer weights                              |
| `weights_3`               | (64, n_pca)     | Output layer weights                             |
| `biases_0` ... `biases_3` | (layer_size,)   | Bias vectors                                     |
| `pca_components`          | (n_pca, 480)    | PCA basis vectors                                |
| `pca_mean`                | (480,)          | Mean spectrum                                    |
| `input_min`               | (n_params,)     | Input scaling lower bounds                       |
| `input_max`               | (n_params,)     | Input scaling upper bounds                       |
| `flx_slr`                 | (480,)          | Solar flux spectrum                              |
| `metadata`                | scalar          | JSON string with param_names, bounds, build info |

## Accuracy and Limitations

- **Expected R²**: >0.999 with 5000+ training samples for typical glacier ice configurations
- **Extrapolation warning**: the emulator should not be used outside its training bounds. `predict()` clips inputs and warns if out of bounds.
- **PCA artefacts**: extreme parameter combinations (very high impurity loading) may produce small spectral artefacts. Increase `n_samples` to mitigate.
- **No RT-solver-only outputs**: `heat_rt` and `absorbed_flux_per_layer` are not available from the emulator (only from the full forward model).

## See Also

- [examples/04_emulator_build.py](../examples/04_emulator_build.py) — building a custom emulator
- [examples/05_emulator_predict.py](../examples/05_emulator_predict.py) — predictions and speed comparison
- [examples/06_emulator_save_load.py](../examples/06_emulator_save_load.py) — save/load and metadata inspection
- [docs/INVERSION.md](INVERSION.md) — using the emulator for parameter retrieval
