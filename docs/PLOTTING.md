# Built-in Plotting

BioSNICAR includes four built-in plotting functions that cover the most common
visualisation tasks: spectral albedo, subsurface light fields, inversion
results, and parameter sensitivity.  All functions share a consistent interface,
return matplotlib objects for further customisation, and can save directly to
disc.

## Installation

Plotting requires `matplotlib`, which is an optional dependency:

```bash
pip install matplotlib
```

Matplotlib is imported lazily — the plotting module has no effect on import time
and the rest of BioSNICAR works without it.


## Quick start

Every plotting function is available both as a **method** on the relevant result
object and as a **standalone function** in `biosnicar.plotting`.  The method
form is the most convenient:

```python
from biosnicar import run_model
from biosnicar.drivers.sweep import parameter_sweep

# Spectral albedo
outputs = run_model(solzen=50, rds=1000)
outputs.plot(show=True)

# Subsurface PAR
outputs.plot_subsurface(save="par_profile.png")

# Parameter sensitivity
df = parameter_sweep(params={"rds": [200, 500, 1000, 2000]})
df.plot_sensitivity(show=True)
```

For inversions:

```python
from biosnicar.inverse.optimize import retrieve

result = retrieve(observed=obs, parameters=[...], emulator=emu)
result.plot(true_values={"rds": 800, "rho": 500}, show=True)
```


## Common parameters

Every plotting function accepts these keyword arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save` | `str` or `Path` | `None` | Save the figure to this file path. Format is inferred from the extension (`.png`, `.pdf`, `.svg`, etc.). |
| `show` | `bool` | `False` | Display the figure in an interactive matplotlib window (`plt.show()`). |
| `dpi` | `int` | `300` | Resolution for saved figures. |
| `figsize` | `tuple` | varies | Figure dimensions in inches `(width, height)`. |

**Behaviour of `save` and `show`:**

- If only `save` is given, the figure is saved and then closed (freeing memory).
- If only `show` is given, the figure is displayed interactively.
- If both are given, the figure is saved first, then displayed.
- If neither is given, the figure remains open in memory and is returned for
  further manipulation.

**Return value:** all functions return `(fig, axes)` — a matplotlib `Figure` and
one or more `Axes` objects.  This lets you adjust titles, add annotations, change
limits, or composite multiple plots before saving:

```python
fig, ax = outputs.plot()
ax.set_title("My custom title")
ax.axhline(0.5, color="red", linestyle="--", label="Threshold")
ax.legend()
fig.savefig("custom_plot.png", dpi=150)
```


---

## 1. `plot_albedo` — Spectral albedo

Plot 480-band spectral albedo from one or more model runs, with optional
satellite band overlays.

### Method form

```python
outputs.plot()
outputs.plot(platform="sentinel2", save="albedo.png")
```

To overlay multiple runs:

```python
clean = run_model(rds=1000)
dirty = run_model(rds=1000, black_carbon=100)
clean.plot(dirty, labels=["Clean", "100 ppb BC"], show=True)
```

### Standalone form

```python
from biosnicar.plotting import plot_albedo

fig, ax = plot_albedo(clean, dirty,
                      labels=["Clean", "100 ppb BC"],
                      platform="sentinel2",
                      xlim=(0.3, 2.5),
                      save="comparison.png")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `*outputs_list` | `Outputs` | *(required)* | One or more `Outputs` objects to plot. |
| `labels` | `list[str]` | auto | Legend labels. Defaults to `None` for a single run, or `"Run 1"`, `"Run 2"`, etc. for multiple. |
| `platform` | `str` | `None` | Overlay convolved band values as horizontal bars. Supported: `"sentinel2"`, `"landsat8"`, `"modis"`. |
| `xlim` | `tuple` | `(0.3, 2.5)` | Wavelength axis limits in micrometres. |
| `title` | `str` | `None` | Plot title. |
| `colors` | `list` | `tab10` | Line colours (one per `Outputs` object). |

### Platform band overlay

When `platform` is specified, the function convolves the spectral albedo from
the *last* `Outputs` object onto the platform's spectral response functions
and draws each band as a horizontal bar centred on the band's wavelength.
This gives an immediate visual comparison between the high-resolution model
spectrum and what a satellite would observe.

Band centres and widths for the overlay are approximate — actual convolution
uses the full spectral response functions from `biosnicar.bands`.

### Example output

```python
outputs = run_model(solzen=50, rds=500, glacier_algae=5000)
outputs.plot(platform="sentinel2", title="Algal ice", show=True)
```

This produces a spectral curve (0.3--2.5 um) with orange horizontal bars at
each Sentinel-2 band position showing the SRF-convolved albedo.


---

## 2. `plot_subsurface` — Subsurface PAR profiles

Two-panel figure showing how Photosynthetically Active Radiation (400--700 nm)
attenuates with depth through the ice column.  This is relevant for biological
habitat studies: the PAR profile determines where within the ice enough light
is available for photosynthesis.

### Method form

```python
outputs = run_model(solzen=50, rds=1000)
outputs.plot_subsurface(show=True)
outputs.plot_subsurface(irradiance=800, save="par_profile.pdf")
```

### Standalone form

```python
from biosnicar.plotting import plot_subsurface

fig, (ax_norm, ax_abs) = plot_subsurface(outputs, irradiance=1000,
                                          save="par.png")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `outputs` | `Outputs` | *(required)* | Must have subsurface flux data populated (i.e. from the adding-doubling solver). |
| `irradiance` | `float` | `1000.0` | Total incoming solar irradiance in W m-2, used to scale normalised model fluxes to absolute PAR in panel (b). |

The default `figsize` is `(12, 5)`.

### What it shows

**Panel (a) — Normalised PAR:**

- **x-axis:** PAR normalised to the surface value, so that 1.0 corresponds to
  the incoming PAR at the surface.
- **y-axis:** depth in centimetres, increasing downward.
- **Dashed grey line:** the surface reference (PAR = 1.0).

Values > 1.0 near the surface are physically real — in a highly scattering,
low-absorption medium (ice in the visible has single-scattering albedo
~0.99998), backscattered light from below augments the downwelling stream.
This radiation trapping effect is strongest within the first transport
mean-free-path and diminishes with depth.

**Panel (b) — Absolute PAR:**

- **x-axis:** PAR in W m-2.
- **y-axis:** depth in centimetres, increasing downward.

The model works in normalised flux units internally (total incoming sums to 1).
Panel (b) converts to absolute units by multiplying by the `irradiance`
parameter.  The default of 1000 W m-2 is a reasonable clear-sky approximation;
set this to your site-specific value for quantitative use.  The assumed
irradiance is annotated on the plot.

### Requirements

The `Outputs` object must have `F_up` and `F_dwn` populated.  This happens
automatically when using the adding-doubling solver (the default).  If you
see a `RuntimeError` about missing subsurface flux data, ensure you are not
using the Toon solver, which does not store interface fluxes.


---

## 3. `plot_retrieval` — Inversion results

Visualise the output of a spectral or band-mode retrieval from the inverse
module.  Produces a two-panel figure: spectral fit (left) and retrieved
parameter values with uncertainties (right).

### Method form

```python
result = retrieve(observed=obs, parameters=["rds", "rho"], emulator=emu)
result.plot(show=True)
result.plot(true_values={"rds": 800, "rho": 500}, save="retrieval.png")
```

### Standalone form

```python
from biosnicar.plotting import plot_retrieval

fig, (ax_spec, ax_params) = plot_retrieval(
    result,
    true_values={"rds": 800, "rho": 500},
    wvl_range=(0.35, 2.5),
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `RetrievalResult` | *(required)* | Output from `retrieve()`. |
| `true_values` | `dict` | `None` | `{param_name: true_value}` for comparison. Shown as black **x** markers. |
| `wvl_range` | `tuple` | `(0.35, 2.5)` | Wavelength range for the spectral panel (only used in spectral mode). |

### Panel (a): Spectral fit

The function automatically detects whether the retrieval used full spectral
observations (480 bands) or satellite band observations:

- **Spectral mode** (>= 100 observations): plots observed spectrum in grey
  and the retrieved fit in blue, with an inset panel showing the residual
  (predicted minus observed).
- **Band mode** (< 100 observations): plots side-by-side bars for observed
  and retrieved values at each band index.

### Panel (b): Retrieved parameters

Horizontal bar chart of the best-fit parameter values with 1-sigma error bars
derived from the Hessian (for gradient-based methods) or the posterior (for
MCMC).  If `true_values` is provided, the true values are overlaid as black
cross markers, allowing immediate visual assessment of retrieval accuracy.


---

## 4. `plot_sensitivity` — Parameter sweep visualisation

Visualise how model outputs respond to changes in input parameters.  This
function takes the output of `parameter_sweep()` and automatically selects
the appropriate plot type based on how many parameters were swept.

### Method form

```python
from biosnicar.drivers.sweep import parameter_sweep

# One parameter swept → line plot
df = parameter_sweep(params={"rds": [100, 200, 500, 1000, 2000]})
df.plot_sensitivity(show=True)
df.plot_sensitivity(y="BBAVIS", save="vis_sensitivity.png")

# Two parameters swept → heatmap or multi-line
df = parameter_sweep(params={
    "rds": [200, 500, 1000, 2000],
    "rho": [400, 500, 600, 700, 800],
})
df.plot_sensitivity(show=True)
```

### Standalone form

```python
from biosnicar.plotting import plot_sensitivity

fig, ax = plot_sensitivity(df, y="BBA", save="sensitivity.png")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sweep_df` | `SweepResult` or `DataFrame` | *(required)* | Output from `parameter_sweep()`. |
| `y` | `str` | `"BBA"` | Output column to plot. Options include `"BBA"`, `"BBAVIS"`, `"BBANIR"`, `"abs_slr_tot"`, `"abs_vis_tot"`, `"abs_nir_tot"`. |

### Automatic plot type selection

The function inspects which columns vary (have more than one unique value) and
chooses:

| Swept params | Both have >= 4 values? | Plot type |
|:---:|:---:|---|
| 1 | -- | **Line plot** with markers: parameter on x-axis, output on y-axis. |
| 2 | Yes | **Heatmap** (`pcolormesh`): first parameter on x, second on y, output as colour. |
| 2 | No | **Multi-line plot**: first parameter on x-axis, second parameter as line colour (hue). |
| 3+ | -- | **Multi-line plot**: first parameter on x-axis, second as hue (remaining parameters are present but not explicitly grouped). |

### Examples

**1D sweep** — how does broadband albedo change with grain size?

```python
df = parameter_sweep(params={"rds": [100, 200, 500, 1000, 2000, 5000]})
df.plot_sensitivity(save="rds_sensitivity.png")
```

**2D sweep** — grain size vs solar zenith angle:

```python
df = parameter_sweep(params={
    "rds": [200, 500, 1000, 2000, 5000],
    "solzen": [30, 40, 50, 60, 70],
})
df.plot_sensitivity(save="rds_solzen_heatmap.png")
```

**2D sweep targeting visible albedo:**

```python
df = parameter_sweep(params={
    "rds": [200, 500, 1000, 2000],
    "black_carbon": [0, 10, 50, 100],
})
df.plot_sensitivity(y="BBAVIS", show=True)
```


---

## Customising plots

All functions return `(fig, axes)`, so you can modify the figure after
creation.  Some common customisations:

### Change axis limits and labels

```python
fig, ax = outputs.plot()
ax.set_xlim(0.4, 1.0)   # zoom into visible range
ax.set_ylim(0, 1)
ax.set_ylabel("Surface albedo")
fig.savefig("zoomed.png")
```

### Add annotations

```python
fig, ax = outputs.plot()
ax.annotate("Ice absorption\nfeature", xy=(1.03, 0.2),
            xytext=(1.2, 0.4), fontsize=9,
            arrowprops=dict(arrowstyle="->"))
fig.savefig("annotated.png")
```

### Composite multiple plot types

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Use standalone functions with existing axes (by modifying the returned figure)
# Or generate separate figures and compose manually:
fig1, ax_alb = outputs.plot()
fig2, ax_par = outputs.plot_subsurface()

# Copy data into a combined figure, or save separately and combine in a
# document.
```

### Use a different matplotlib style

```python
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
outputs.plot(show=True)
```


---

## Summary

| Function | Method on | Description | Return |
|----------|-----------|-------------|--------|
| `plot_albedo` | `Outputs.plot()` | Spectral albedo with optional platform band overlay | `fig, ax` |
| `plot_subsurface` | `Outputs.plot_subsurface()` | Normalised + absolute PAR depth profiles | `fig, (ax_norm, ax_abs)` |
| `plot_retrieval` | `RetrievalResult.plot()` | Spectral fit + retrieved parameters with uncertainties | `fig, (ax_spec, ax_params)` |
| `plot_sensitivity` | `SweepResult.plot_sensitivity()` | Auto-detected line / heatmap / multi-line sensitivity plot | `fig, ax` |

All functions accept `save`, `show`, `dpi`, and `figsize`.  All return
matplotlib objects for further customisation.
