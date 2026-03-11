# Platform Band Convolution

BioSNICAR outputs 480-band spectral albedo at 10 nm resolution
(0.205--4.995 um). Satellites and climate models observe the surface through
coarser spectral windows. The `biosnicar.bands` module maps the
high-resolution model spectrum onto platform-specific bands, enabling direct
comparison between simulated and observed albedos.

**Supported platforms:**

| Platform    | Key       | Method            | Bands | Indices |
|-------------|-----------|-------------------|------:|---------|
| Sentinel-2  | `sentinel2`  | SRF convolution | 13 | NDSI, NDVI, II |
| Sentinel-3  | `sentinel3`  | SRF convolution | 21 | I2DBA, I3DBA, NDCI, MCI, NDVI |
| Landsat 8   | `landsat8`   | SRF convolution | 7  | NDSI, NDVI |
| MODIS       | `modis`      | Interval average | 7 + 3 broadband | NDSI |
| CESM 2-band | `cesm2band`  | Interval average | 2  | -- |
| CESM RRTMG  | `cesmrrtmg`  | Interval average | 14 | -- |
| MAR         | `mar`        | Interval average | 4  | -- |
| HadCM3      | `hadcm3`     | Interval average | 6  | -- |

---

## Quick start

`.to_platform()` chains directly onto `run_model()`:

```python
from biosnicar import run_model

# Run the forward model and convolve onto Sentinel-2
s2 = run_model(solzen=50, rds=1000).to_platform("sentinel2")
print(s2.B3)        # green-band albedo
print(s2.NDSI)      # Normalized Difference Snow Index

# Same for a GCM
cesm = run_model(solzen=50, rds=1000).to_platform("cesm2band")
print(cesm.vis)     # VIS broadband albedo
print(cesm.nir)     # NIR broadband albedo
```

A single `Outputs` object can be convolved onto multiple platforms:

```python
outputs = run_model(solzen=50, rds=1000)
for plat in ["sentinel2", "modis", "cesm2band", "mar"]:
    print(outputs.to_platform(plat))
```

For parameter sweeps, `.to_platform()` appends band columns to the
DataFrame:

```python
from biosnicar.drivers.sweep import parameter_sweep

df = parameter_sweep(
    params={"rds": [500, 1000], "solzen": [50, 60]},
).to_platform("sentinel2")

print(df[["rds", "solzen", "BBA", "B3", "NDSI"]])
```

---

## API reference

### `Outputs.to_platform(platform)` (chaining from `run_model()`)

The preferred way to convolve a single model run. Chains directly onto the
`Outputs` object returned by `run_model()`:

```python
s2 = run_model(solzen=50).to_platform("sentinel2")
```

| Parameter  | Type  | Description |
|------------|-------|-------------|
| `platform` | `str` | Platform key (see table above). |

**Returns:** a `BandResult` object.

### `SweepResult.to_platform(*platforms)` (chaining from `parameter_sweep()`)

Chains onto the `SweepResult` returned by `parameter_sweep()`, applying the
convolution to every row and appending band/index columns to the DataFrame:

```python
df = parameter_sweep(params={...}).to_platform("sentinel2")
df = parameter_sweep(params={...}).to_platform("sentinel2", "modis")
```

| Parameter    | Type    | Description |
|--------------|---------|-------------|
| `*platforms` | `str`   | One or more platform keys. With a single platform, columns are unprefixed (`B3`, `NDSI`). With multiple platforms, columns are prefixed (`sentinel2_B3`, `modis_B1`). |

**Returns:** a `pandas.DataFrame` with the original sweep columns plus band
albedo and index columns.

### `to_platform(albedo, platform, flx_slr=None)` (standalone)

The standalone function is available for cases where you have raw arrays
(e.g. from custom pipelines or external data):

```python
from biosnicar.bands import to_platform

result = to_platform(albedo_array, "sentinel2", flx_slr=flux_array)
```

| Parameter  | Type              | Description |
|------------|-------------------|-------------|
| `albedo`   | `np.ndarray (480,)` | Spectral albedo. |
| `platform` | `str`             | Platform key. |
| `flx_slr`  | `np.ndarray (480,)` | Spectral solar flux. Required for flux-weighted convolution. |

**Returns:** a `BandResult` object.

Also available at package level as `biosnicar.to_platform()`.

### `BandResult`

Container for convolved band albedos and spectral indices. Attributes are
set dynamically by each platform module.

| Attribute     | Type          | Description |
|---------------|---------------|-------------|
| `platform`    | `str`         | Platform key. |
| `band_names`  | `list[str]`   | Names of all band attributes (e.g. `["B1", "B2", ...]`). |
| `index_names` | `list[str]`   | Names of all index attributes (e.g. `["NDSI", "NDVI"]`). |

Band and index values are set as named attributes, so `result.B3` returns
the B3 band albedo directly.

**Methods:**

- `as_dict()` -- returns a flat `dict` of all bands and indices.
- `repr()` -- pretty-prints all band and index values.

Example output of `print(result)`:

```
BandResult(platform='sentinel2')
  Bands:
    B1: 0.4907
    B2: 0.4898
    ...
  Indices:
    NDSI: 0.8877
    NDVI: -0.1355
    II: 1.0655
```

---

## Use cases

### Forward modelling for retrieval validation

Predict what a satellite would observe for a given ice configuration:

```python
from biosnicar import run_model

s2 = run_model(solzen=50, rds=500, black_carbon=1000).to_platform("sentinel2")
print(f"Sentinel-2 B3 (green): {s2.B3:.3f}")
print(f"Sentinel-2 NDSI:       {s2.NDSI:.3f}")
```

### Spectral index calibration

Compute spectral indices in native satellite band space across a range of
surface conditions, using `parameter_sweep`:

```python
from biosnicar.drivers.sweep import parameter_sweep

df = parameter_sweep(
    params={"rds": [100, 250, 500, 1000, 2000]},
).to_platform("sentinel2")

print(df[["rds", "BBA", "B3", "NDSI"]])
```

### GCM parameterisation

Generate band albedos for a climate model's radiation scheme:

```python
from biosnicar import run_model

cesm = run_model(solzen=60, rds=1000).to_platform("cesm2band")
print(f"CESM VIS: {cesm.vis:.4f}, NIR: {cesm.nir:.4f}")

mar = run_model(solzen=60, rds=1000).to_platform("mar")
print(f"MAR  sw1: {mar.sw1:.4f}, sw2: {mar.sw2:.4f}, sw3: {mar.sw3:.4f}, sw4: {mar.sw4:.4f}")
```

Or sweep parameter space and get GCM bands directly:

```python
from biosnicar.drivers.sweep import parameter_sweep

df = parameter_sweep(
    params={"rds": [500, 1000, 2000], "solzen": [40, 50, 60]},
).to_platform("cesm2band")

print(df[["rds", "solzen", "BBA", "vis", "nir"]])
```

### Multi-platform comparison

Compare the same surface across several sensors simultaneously using
`parameter_sweep().to_platform()` with multiple platforms:

```python
from biosnicar.drivers.sweep import parameter_sweep

df = parameter_sweep(
    params={"rds": [500, 1000]},
).to_platform("sentinel2", "landsat8", "modis")

print(df[["rds", "sentinel2_NDSI", "landsat8_NDSI", "modis_NDSI"]])
```

Or loop over platforms from a single `Outputs` object:

```python
from biosnicar import run_model

outputs = run_model(solzen=50, rds=1000)
for plat in ["sentinel2", "landsat8", "modis"]:
    r = outputs.to_platform(plat)
    print(f"{plat:12s}  bands: {len(r.band_names):2d}  indices: {r.index_names}")
```

---

## Convolution methods

Two methods are used depending on the platform:

### SRF convolution (satellite platforms)

Used by: Sentinel-2, Sentinel-3, Landsat 8.

Each satellite band has a Spectral Response Function (SRF) describing its
sensitivity across wavelength. The band-averaged albedo is:

```
band_albedo = sum(albedo * SRF * flx_slr) / sum(SRF * flx_slr)
```

SRFs are stored as CSV files in `data/band_srfs/` on the 480-band model
grid. They are loaded once and cached in memory.

### Flux-weighted interval averaging (GCMs and MODIS)

Used by: CESM, MAR, HadCM3, MODIS.

Climate models use fixed wavelength intervals. The band albedo is the
flux-weighted mean over all model grid points within the interval:

```
band_albedo = sum(albedo[lo:hi] * flx_slr[lo:hi]) / sum(flx_slr[lo:hi])
```

MODIS bands are wide enough (20--115 nm) that the precise SRF shape has
minimal effect at 10 nm resolution, so they are also treated as intervals.

---

## Platform details and data provenance

### Sentinel-2 MSI

**Source:** ESA Sentinel-2 MSI Technical Guide. Band centres and widths from
the [Sentinel-2 User Handbook](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)
(ESA, 2015), Table 1.

**SRF file:** `data/band_srfs/sentinel2_msi.csv`

The current SRFs use a **tophat approximation**: response = 1.0 within the
published band edges, 0.0 outside. This can be replaced with the official
ESA spectral response curves (available from the Sentinel-2 document library)
for higher fidelity, by overwriting the CSV.

| Band | Centre (nm) | Width (nm) | Range (nm)  | Description      |
|------|-------------|------------|-------------|------------------|
| B1   | 443         | 20         | 433--453    | Coastal aerosol  |
| B2   | 490         | 65         | 457.5--522.5| Blue             |
| B3   | 560         | 35         | 542.5--577.5| Green            |
| B4   | 665         | 30         | 650--680    | Red              |
| B5   | 705         | 15         | 697.5--712.5| Vegetation red edge |
| B6   | 740         | 15         | 732.5--747.5| Vegetation red edge |
| B7   | 783         | 20         | 773--793    | Vegetation red edge |
| B8   | 842         | 115        | 784.5--899.5| NIR              |
| B8A  | 865         | 20         | 855--875    | Narrow NIR       |
| B9   | 945         | 20         | 935--955    | Water vapour     |
| B10  | 1375        | 30         | 1360--1390  | SWIR -- Cirrus   |
| B11  | 1610        | 90         | 1565--1655  | SWIR             |
| B12  | 2190        | 180        | 2100--2280  | SWIR             |

**Spectral indices:**

| Index | Formula                    | Reference |
|-------|----------------------------|-----------|
| NDSI  | (B3 - B11) / (B3 + B11)   | Hall et al. (1995) |
| NDVI  | (B8 - B4) / (B8 + B4)     | Tucker (1979) |
| II    | B3 / B8A                   | Cook et al. (2020) Impurity Index |

---

### Sentinel-3 OLCI

**Source:** ESA Sentinel-3 OLCI Technical Guide. Band centres and widths from
the [Sentinel-3 OLCI User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-3-olci)
(ESA, 2016), Table 1.

**SRF file:** `data/band_srfs/sentinel3_olci.csv`

Tophat approximation from published band centres and widths. Some very
narrow bands (Oa13: 2.5 nm, Oa15: 2.5 nm) are narrower than the 10 nm model
grid and are represented by a single grid point nearest to the band centre.
Bands Oa13, Oa14, and Oa15 are all mapped to the 765 nm grid point --
this is inherent to the 10 nm model resolution and means these O2 A-band
channels cannot be distinguished in convolved output.

| Band  | Centre (nm) | Width (nm) | Description               |
|-------|-------------|------------|---------------------------|
| Oa01  | 400.0       | 15.0       | Aerosol correction        |
| Oa02  | 412.5       | 10.0       | Yellow substance           |
| Oa03  | 442.5       | 10.0       | Chl absorption max         |
| Oa04  | 490.0       | 10.0       | Chl, other pigments        |
| Oa05  | 510.0       | 10.0       | Chl, sediment, turbidity   |
| Oa06  | 560.0       | 10.0       | Chl reference              |
| Oa07  | 620.0       | 10.0       | Sediment loading           |
| Oa08  | 665.0       | 10.0       | Chl (2nd max), sediment    |
| Oa09  | 673.75      | 7.5        | Chl fluorescence           |
| Oa10  | 681.25      | 7.5        | Chl fluorescence peak      |
| Oa11  | 708.75      | 10.0       | Chl fluorescence, red edge |
| Oa12  | 753.75      | 10.0       | O2 reference               |
| Oa13  | 761.25      | 2.5        | O2 A-band                  |
| Oa14  | 764.375     | 3.75       | Atmospheric correction     |
| Oa15  | 767.5       | 2.5        | O2 A-band                  |
| Oa16  | 778.75      | 15.0       | Atm. correction, aerosol   |
| Oa17  | 865.0       | 20.0       | Atm. correction, aerosol   |
| Oa18  | 885.0       | 10.0       | Water vapour               |
| Oa19  | 900.0       | 10.0       | Water vapour               |
| Oa20  | 940.0       | 20.0       | Water vapour               |
| Oa21  | 1020.0      | 40.0       | Atm. correction            |

**Spectral indices:**

| Index  | Formula                                   | Reference |
|--------|-------------------------------------------|-----------|
| I2DBA  | Oa12 / Oa08                               | Wang et al. (2020) 2-band diagnostic absorption |
| I3DBA  | (Oa08 - Oa11) / Oa13                      | Wang et al. (2020) 3-band diagnostic absorption |
| NDCI   | (Oa11 - Oa08) / (Oa11 + Oa08)            | Mishra & Mishra (2012) |
| MCI    | Oa11 - Oa10 - (Oa12 - Oa10) * 0.379      | Gower et al. (2005) Maximum Chlorophyll Index |
| NDVI   | (Oa17 - Oa08) / (Oa17 + Oa08)            | Tucker (1979) |

The MCI uses linear baseline interpolation between Oa10 (681.25 nm) and
Oa12 (753.75 nm) evaluated at Oa11 (708.75 nm):
`factor = (708.75 - 681.25) / (753.75 - 681.25) = 0.379`.

---

### Landsat 8 OLI

**Source:** USGS Landsat 8 Data Users Handbook, Version 5.0 (USGS, 2019),
Table 5-1.

**SRF file:** `data/band_srfs/landsat8_oli.csv`

Tophat approximation from published band edges. The official OLI relative
spectral response curves are available from the [USGS Landsat Calibration
page](https://www.usgs.gov/landsat-missions/landsat-calibration-and-validation)
and can replace the CSV for higher fidelity.

| Band | Centre (nm) | Range (nm)   | Description      |
|------|-------------|--------------|------------------|
| B1   | 443         | 435--451     | Coastal aerosol  |
| B2   | 482         | 452--512     | Blue             |
| B3   | 561         | 533--590     | Green            |
| B4   | 655         | 636--673     | Red              |
| B5   | 865         | 851--879     | NIR              |
| B6   | 1609        | 1567--1651   | SWIR-1           |
| B7   | 2201        | 2107--2294   | SWIR-2           |

**Spectral indices:**

| Index | Formula                    | Reference |
|-------|----------------------------|-----------|
| NDSI  | (B3 - B6) / (B3 + B6)     | Hall et al. (1995) |
| NDVI  | (B5 - B4) / (B5 + B4)     | Tucker (1979) |

---

### MODIS (Terra / Aqua)

**Source:** MODIS Level 1 Calibration documentation (Xiong et al., 2009).
Band definitions from the [MODIS Specifications page](https://modis.gsfc.nasa.gov/about/specifications.php).

**Method:** Interval averaging (no SRF CSV file needed).

The seven MODIS land bands are wide enough (20--115 nm) that the precise SRF
shape has negligible effect at 10 nm model resolution. The module uses
flux-weighted interval averaging rather than SRF convolution.

| Band | Centre (nm) | Range (nm)   | Description      |
|------|-------------|--------------|------------------|
| B1   | 645         | 620--670     | Land/cloud/aerosol boundaries |
| B2   | 858         | 841--876     | Land/cloud/aerosol boundaries |
| B3   | 469         | 459--479     | Land/cloud/aerosol properties |
| B4   | 555         | 545--565     | Land/cloud/aerosol properties |
| B5   | 1240        | 1230--1250   | Land/cloud/aerosol properties |
| B6   | 1640        | 1628--1652   | Land/cloud/aerosol properties |
| B7   | 2130        | 2105--2155   | Land/cloud/aerosol properties |

Three broadband aggregations are also computed:

| Name | Range (um) | Description |
|------|-----------|-------------|
| VIS  | 0.3--0.7  | Visible broadband |
| NIR  | 0.7--5.0  | Near-infrared broadband |
| SW   | 0.3--5.0  | Full shortwave broadband |

**Spectral index:**

| Index | Formula                    | Reference |
|-------|----------------------------|-----------|
| NDSI  | (B4 - B6) / (B4 + B6)     | Hall et al. (2002) |

---

### CESM 2-band

**Source:** Community Earth System Model (CESM) shortwave radiation
scheme. The 2-band VIS/NIR split is the standard surface albedo
representation in CESM's coupler (Oleson et al., 2013, CLM Technical Note).

**Method:** Interval averaging.

| Band | Range (um) | Description |
|------|-----------|-------------|
| vis  | 0.2--0.7  | Visible     |
| nir  | 0.7--5.0  | Near-infrared |

These are the two albedo bands exchanged between the land model (CLM) and
the atmosphere model (CAM) at each coupling timestep.

---

### CESM RRTMG (14-band shortwave)

**Source:** RRTMG shortwave radiation scheme as implemented in CAM
(Iacono et al., 2008, J. Geophys. Res., 113, D13103). Band boundaries
from Table 1 of the paper.

**Method:** Interval averaging.

| Band | Range (um)      | Wavenumber (cm-1) |
|------|-----------------|-------------------|
| sw1  | 3.077--3.846    | 2600--3250        |
| sw2  | 2.500--3.077    | 3250--4000        |
| sw3  | 2.151--2.500    | 4000--4650        |
| sw4  | 1.942--2.151    | 4650--5150        |
| sw5  | 1.626--1.942    | 5150--6150        |
| sw6  | 1.299--1.626    | 6150--7700        |
| sw7  | 1.242--1.299    | 7700--8050        |
| sw8  | 0.778--1.242    | 8050--12850       |
| sw9  | 0.625--0.778    | 12850--16000      |
| sw10 | 0.442--0.625    | 16000--22650      |
| sw11 | 0.345--0.442    | 22650--29000      |
| sw12 | 0.263--0.345    | 29000--38000      |
| sw13 | 0.200--0.263    | 38000--50000      |
| sw14 | 3.846--12.195   | 820--2600         |

Band sw14 extends well beyond the BioSNICAR model range (which ends at
4.995 um), so the returned albedo for sw14 represents only the portion
within the model grid.

---

### MAR (4-band shortwave)

**Source:** MAR regional climate model shortwave radiation scheme
(Fettweis et al., 2017, The Cryosphere, 11, 517-532). Band definitions
from the MAR radiation documentation; see also Brun et al. (1992) for the
original Crocus/ISBA scheme that MAR inherits.

**Method:** Interval averaging.

| Band | Range (um)  | Description |
|------|------------|-------------|
| sw1  | 0.25--0.69 | Visible     |
| sw2  | 0.69--1.19 | Near-IR     |
| sw3  | 1.19--2.38 | Shortwave-IR |
| sw4  | 2.38--4.00 | Mid-IR      |

---

### HadCM3 (6-band Edwards-Slingo)

**Source:** Edwards-Slingo shortwave radiation scheme as used in the
Met Office Unified Model and HadCM3 (Edwards & Slingo, 1996, Q. J. R.
Meteorol. Soc., 122, 689-719). Band boundaries from Table 2 of the paper.

**Method:** Interval averaging.

| Band | Range (um)  | Description |
|------|------------|-------------|
| es1  | 0.20--0.32 | UV          |
| es2  | 0.32--0.69 | Visible     |
| es3  | 0.69--1.19 | Near-IR     |
| es4  | 1.19--2.38 | Shortwave-IR |
| es5  | 2.38--4.00 | Mid-IR      |
| es6  | 4.00--5.00 | Far-IR (clipped to model range) |

Band es6 extends to 10 um in the original scheme but is clipped to the
BioSNICAR model ceiling of 5.0 um.

---

## SRF data files

Satellite spectral response functions are stored in `data/band_srfs/` as
CSV files with the following format:

```csv
wavelength_um,B1,B2,B3,...
0.205,0.0,0.0,0.0,...
0.215,0.0,0.0,0.0,...
...
4.995,...
```

Each file has 480 rows (one per model wavelength) and one column per band.
Values are between 0 and 1.

**Current files:**

| File                    | Platform    | Bands | SRF type |
|-------------------------|-------------|------:|----------|
| `sentinel2_msi.csv`    | Sentinel-2  | 13    | Tophat   |
| `sentinel3_olci.csv`   | Sentinel-3  | 21    | Tophat   |
| `landsat8_oli.csv`     | Landsat 8   | 7     | Tophat   |

### Replacing with official SRFs

The initial tophat approximations can be replaced with manufacturer-provided
spectral response curves for higher fidelity. To do so:

1. Obtain the official SRF data (e.g. from ESA's Sentinel document library
   or USGS Landsat calibration page).
2. Interpolate the SRF onto the 480-band model grid
   (0.205, 0.215, ..., 4.995 um).
3. Normalise so the peak value is 1.0.
4. Write the CSV in the same format as the existing files, preserving the
   column names.
5. Overwrite the corresponding file in `data/band_srfs/`.

The module will load the new SRFs automatically on next use (clear any
running Python process to reset the in-memory cache).

---

## Implementation notes

### How `flx_slr` flows through the API

The `run_model()` pipeline populates `outputs.flx_slr` -- the 480-band
spectral solar flux used by the radiative transfer solver. This is the same
flux array used internally to compute broadband albedo (`outputs.BBA`). Both
the adding-doubling and Toon solvers copy this array onto the `Outputs`
object, so it is always available after a model run.

When chaining `run_model().to_platform()`, the `Outputs.to_platform()`
method passes `self.albedo` and `self.flx_slr` to the convolution
automatically -- the user never needs to handle raw arrays.

When chaining `parameter_sweep().to_platform()`, the `SweepResult` stores
spectral data internally for each row (regardless of the `return_spectral`
flag), so `.to_platform()` always has what it needs.

### Flux weighting

All convolutions are flux-weighted: the albedo in each band is weighted by
the incoming solar flux within that band. This is physically correct because
a satellite or GCM "sees" the surface weighted by the illumination spectrum.
Without flux weighting, bands in low-flux regions (e.g. beyond 2.5 um)
would have disproportionate influence.

### Caching

SRF CSV files are loaded once per Python session and cached in a module-level
dictionary. There is no disk cache -- the files are small (~50 KB each) and
load in under 1 ms.

### Limitations

- **10 nm resolution:** The model grid is 10 nm, so satellite bands narrower
  than 10 nm (Sentinel-3 Oa13, Oa14, Oa15) are represented by a single grid
  point. The O2 A-band structure cannot be resolved.
- **Tophat SRFs:** The initial SRF files use rectangular (tophat) response
  functions. For most cryospheric applications the error is small, but users
  requiring high radiometric accuracy should substitute official SRFs (see
  above).
- **Model wavelength ceiling:** BioSNICAR extends to 4.995 um. RRTMG band
  sw14 (3.846--12.195 um) and HadCM3 band es6 (4.0--10.0 um) are truncated
  at the model ceiling. Flux beyond 5 um is negligible for shortwave surface
  albedo.

---

## Testing

Run the band-convolution test suite:

```
python -m pytest tests/test_bands.py -v
```

The tests are divided into:

- **Core utilities:** wavelength grid shape, interval averaging, SRF
  convolution with known inputs.
- **Platform tests (unit):** flat albedo (= 0.5) should return 0.5 for all
  bands; flat albedo should produce NDSI = 0; ramp albedo should produce
  physically consistent index signs.
- **Integration tests** (marked `@pytest.mark.slow`): full `run_model()` →
  `to_platform()` pipeline for all eight platforms.

To run only the fast unit tests:

```
python -m pytest tests/test_bands.py -v -k "not slow"
```

To include integration tests:

```
python -m pytest tests/test_bands.py -v
```

---

## References

Brun, E., David, P., Sudul, M., & Brunot, G. (1992). A numerical model to
simulate snow-cover stratigraphy for operational avalanche forecasting.
Journal of Glaciology, 38(128), 13-22.

Cook, J. M. et al. (2020). Glacier algae accelerate melt rates on the
western Greenland Ice Sheet. The Cryosphere, 14, 309-330.
https://doi.org/10.5194/tc-14-309-2020

Edwards, J. M., & Slingo, A. (1996). Studies with a flexible new radiation
code. I: Choosing a configuration for a large-scale model. Quarterly Journal
of the Royal Meteorological Society, 122(531), 689-719.
https://doi.org/10.1002/qj.49712253107

ESA (2015). Sentinel-2 User Handbook. European Space Agency.
https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi

ESA (2016). Sentinel-3 OLCI User Guide. European Space Agency.
https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-3-olci

Fettweis, X. et al. (2017). Reconstructions of the 1900-2015 Greenland ice
sheet surface mass balance using the regional climate MAR model. The
Cryosphere, 11, 517-532. https://doi.org/10.5194/tc-11-517-2017

Gower, J., King, S., & Borstad, G. (2005). Detection of intense plankton
blooms using the 709 nm band of the MERIS imaging spectrometer.
International Journal of Remote Sensing, 26(9), 2005-2012.

Hall, D. K., Riggs, G. A., & Salomonson, V. V. (1995). Development of
methods for mapping global snow cover using moderate resolution imaging
spectroradiometer data. Remote Sensing of Environment, 54(2), 127-140.

Hall, D. K., Riggs, G. A., Salomonson, V. V., DiGirolamo, N. E., & Bayr,
K. J. (2002). MODIS snow-cover products. Remote Sensing of Environment,
83(1-2), 181-194.

Iacono, M. J. et al. (2008). Radiative forcing by long-lived greenhouse
gases: Calculations with the AER radiative transfer models. Journal of
Geophysical Research, 113, D13103. https://doi.org/10.1029/2008JD009944

Mishra, S., & Mishra, D. R. (2012). Normalized difference chlorophyll index:
A novel model for remote estimation of chlorophyll-a concentration in
turbid productive waters. Remote Sensing of Environment, 117, 394-406.

Oleson, K. W. et al. (2013). Technical description of version 4.5 of the
Community Land Model (CLM). NCAR Technical Note NCAR/TN-503+STR.

Tucker, C. J. (1979). Red and photographic infrared linear combinations for
monitoring vegetation. Remote Sensing of Environment, 8(2), 127-150.

USGS (2019). Landsat 8 Data Users Handbook, Version 5.0. U.S. Geological
Survey. https://www.usgs.gov/landsat-missions/landsat-8-data-users-handbook

Wang, S. et al. (2020). Retrieval of snow and ice surface properties from
OLCI observations. Remote Sensing of Environment, 240, 111698.

Xiong, X. et al. (2009). MODIS on-orbit calibration and characterization.
Metrologia, 46(4), S103.
