# Subsurface Light Field

BioSNICAR's radiative transfer solvers compute spectral upwelling and
downwelling fluxes at every layer interface.  These arrays are now exposed
as first-class `Outputs` attributes, together with convenience methods for
depth-interpolation, PAR estimation, and spectral heating rates.

This allows users to answer questions such as "what is the PAR at 5 cm
depth?" or "how does spectral irradiance change with impurity loading?"
without modifying solver internals.

---

## Quick start

```python
from biosnicar import run_model

outputs = run_model(solzen=60, rds=500, dz=[0.05, 0.5, 0.45])

# Downwelling spectral flux at 3 cm depth
flux = outputs.subsurface_flux(0.03)
print(flux["F_dwn"].shape)   # (480,)

# PAR (400-700 nm) at the surface
print(outputs.par(0.0))       # fraction of total incoming in PAR range

# PAR at several depths
print(outputs.par([0.0, 0.05, 0.1, 0.5]))
```

---

## API reference

### Raw flux arrays

After a call to `run_model()`, the returned `Outputs` object carries:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `F_up`    | `(nbr_wvl, nbr_lyr+1)` | Spectral upwelling flux at each layer interface. |
| `F_dwn`   | `(nbr_wvl, nbr_lyr+1)` | Spectral downwelling flux at each layer interface. |

Interface indexing: column 0 is the **surface** (top of column), column
`nbr_lyr` is the **bottom** (base of the deepest layer).

**Units:** Fluxes are normalised so that total incoming irradiance sums to 1
across all 480 bands.  To obtain absolute W/m² values, multiply by the
actual total incoming irradiance for your site.

Both the adding-doubling and Toon solvers populate these arrays.

### `outputs.subsurface_flux(depth_m)`

Returns spectral fluxes at an arbitrary depth by linear interpolation
between the two bracketing layer interfaces.

| Parameter | Type | Description |
|-----------|------|-------------|
| `depth_m` | `float` or `array-like` | Depth below surface in metres. |

**Returns:** `dict` with keys `'F_up'`, `'F_dwn'`, `'F_net'`.

- Scalar depth: each value has shape `(nbr_wvl,)`.
- Array depth: each value has shape `(len(depth_m), nbr_wvl)`.

Depths are clipped to the column bounds: `depth=0` returns interface 0,
`depth >= total_thickness` returns the bottom interface.

### `outputs.par(depth_m=0.0)`

Photosynthetically Active Radiation (400--700 nm) at a given depth.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth_m` | `float` or `array-like` | `0.0` | Depth in metres. |

**Returns:** scalar for scalar depth, 1-D array for array depth.

PAR is the sum of downwelling planar irradiance over the 400--700 nm
bands.

**Flux enhancement near the surface:** In a highly scattering,
low-absorption medium (ice in the visible has ssa ~0.99998),
backscattered light from below adds to the downwelling stream, so PAR
just below the surface can *exceed* the incoming surface value. This is
real physics — "radiation trapping" — not a numerical artefact. Photons
in the visible scatter many times before being absorbed, so the local
irradiance within the medium is enhanced relative to the boundary
value. The enhancement is strongest in the top ~1 transport
mean-free-path and diminishes with depth as absorption gradually
removes energy from the field.

### `outputs.spectral_heating_rate()`

Spectral radiative heating rate per layer.

**Returns:** array of shape `(nbr_wvl, nbr_lyr)` in units of K/hr per band.

Computed from:

```
F_net = F_up - F_dwn
F_abs = F_net[:, 1:] - F_net[:, :-1]
heating_rate = F_abs / (L_snw * 2117) * 3600
```

**Note on spectral distribution:** Heating rates are typically dominated
by NIR wavelengths, not visible ones.  Ice is highly transparent in the
visible (absorption coefficient is negligible from 400--700 nm), so VIS
photons scatter through the column without depositing much energy.  In
the NIR, ice absorption increases by orders of magnitude — especially
near 1.0, 1.25, and 1.5 µm — so these wavelengths are absorbed
efficiently in the upper layers and dominate radiative heating.

where `L_snw` is the mass of ice in each layer (kg/m²) and 2117 is the
specific heat capacity of ice (J kg⁻¹ K⁻¹).

---

## Worked example: clean vs algae-loaded PAR profile

```python
import numpy as np
from biosnicar import run_model

depths = np.linspace(0, 0.5, 20)

clean = run_model(solzen=50, rds=500, dz=[0.01, 0.49])
dirty = run_model(solzen=50, rds=500, dz=[0.01, 0.49], glacier_algae=50000)

par_clean = clean.par(depths)
par_dirty = dirty.par(depths)

for d, pc, pd in zip(depths, par_clean, par_dirty):
    print(f"  {d:.3f} m   clean={pc:.4f}  algae={pd:.4f}")
```

---

## Flux enhancement near the surface

Both `F_dwn` and `F_up` at sub-surface interfaces can **exceed** the
incoming surface values.  This is not a numerical artefact — it is a real
physical effect called *radiation trapping* (or flux enhancement).

At the surface (interface 0), `F_dwn` is purely the incoming radiation:
there is no scattering medium above to redirect light back downward.
At the first sub-surface interface, `F_dwn` has three components:

1. Direct beam transmitted through the overlying layer (slightly attenuated).
2. Diffuse sky radiation transmitted through the overlying layer.
3. Upwelling radiation from below that is **scattered back downward** by
   the overlying layer.

In the visible, where ice has an extremely high single-scattering albedo
(ssa ~0.99998), contribution (3) more than compensates for the small
attenuation of components (1) and (2).  Photons bounce many times before
being absorbed, so both hemispheric flux streams are enhanced within the
medium relative to the boundary.  This is the same physics that makes deep
snow glow from within when you dig a snow pit.

The enhancement is strongest in the top ~1 transport mean-free-path and
diminishes with depth as absorption gradually removes energy.  The **net**
flux (`F_dwn - F_up`) always decreases monotonically with depth, as
required by energy conservation.

The effect is most pronounced in the PAR band (400--700 nm) where ice
absorption is negligible, and is weaker in the NIR where absorption is
much stronger.

## Limitations

- **Two-stream approximation:** The solver provides hemispheric (planar)
  fluxes, not angular radiance distributions.
- **Linear interpolation:** `subsurface_flux()` linearly interpolates
  between layer interfaces.  For thick layers the within-layer profile may
  be exponential rather than linear; splitting thick layers into thinner
  sub-layers improves accuracy.
- **Normalised fluxes:** The model normalises total incoming irradiance to
  1.  Multiply outputs by actual irradiance to obtain absolute values.
- **Plane-parallel:** No lateral transport is modelled.

---

## Testing

```
python -m pytest tests/test_subsurface.py -v
```

11 tests cover: array shapes and presence for both solvers, surface albedo
consistency, energy conservation, depth interpolation (boundaries and
mid-layer), PAR magnitude and monotonicity, array input, and spectral
heating rate shape.
