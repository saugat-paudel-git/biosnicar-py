# BioSNICAR Notebooks

Three self-contained teaching notebooks that cover the physics and data
science behind BioSNICAR.  Each notebook builds every concept from first
principles using toy models before connecting to the real codebase, so
they work as standalone educational resources even without BioSNICAR
installed.

## Notebooks

### 1. Radiative Transfer in Snow and Ice (`radiative_transfer_fundamentals.ipynb`)

A physics primer that walks through the complete radiative transfer
pipeline from Maxwell's equations to broadband albedo:

| Sections | Topic |
|----------|-------|
| 1--3 | Mie theory, size parameter, extinction/scattering/absorption efficiencies |
| 4--5 | Ice refractive index, single-scattering albedo, geometric optics limit |
| 6--8 | Grain shape effects, snow vs bubbly ice, liquid water coatings |
| 9--11 | Radiative transfer equation, two-stream approximation, delta-Eddington scaling |
| 12--13 | Single-layer analytical solution, adding method for multilayer media |
| 14 | Light-absorbing particles: black carbon |
| 15--18 | Mineral dust, snow algae, glacier algae |
| 19--22 | Full adding-doubling pipeline, energy conservation, illumination geometry |
| 23--24 | Subsurface light fields, real BioSNICAR forward model (optional) |

**Audience:** Graduate students and researchers who want to understand the
physics underpinning BioSNICAR.  Assumes undergraduate-level optics and
calculus.

### 2. Data Science Foundations for the Emulator and Inversion (`emulator_and_inversions.ipynb`)

A training manual covering every data-science technique used in the
BioSNICAR emulator and inversion scheme:

| Sections | Topic |
|----------|-------|
| 1--4 | Latin hypercube sampling, curse of dimensionality, log-space sampling |
| 5--8 | PCA compression, neural networks from scratch, complete emulator pipeline |
| 9 | Surrogate method comparison: MLP vs GP vs random forest |
| 10--12 | Cost functions, chi-squared, regularisation, log-space optimisation |
| 13--16 | L-BFGS-B, Nelder-Mead, differential evolution, hybrid DE + L-BFGS-B |
| 17--22 | Hessian uncertainty, MCMC / Bayesian inference, error propagation |
| 23--24 | Method comparison, decision guide |
| 25 | Capstone: real BioSNICAR retrieval with spectral and Sentinel-2 band modes (optional) |

**Audience:** Scientists applying BioSNICAR inversions to remote-sensing
data, or anyone building emulator-based retrieval pipelines.  Assumes
basic Python and linear algebra.


### 3. Remote Sensing of Snow and Ice (`remote_sensing_with_biosnicar.ipynb`)

A practitioner's guide for anyone who wants to use BioSNICAR with
satellite data but may have little prior knowledge of radiative transfer
or remote sensing:

| Sections | Topic |
|----------|-------|
| 1--5 | What satellites see: spectra, bands, SRFs, platforms, spectral indices |
| 6--9 | Bridging spectra and bands: forward problem, information content, multi-platform |
| 10--14 | The inverse problem: degeneracy, log-space optimisation, uncertainty, masking |
| 15--18 | Emulators: why they matter, default emulator, custom builds, `retrieve()` API |
| 19--23 | Real-world scenarios: SSA, Sentinel-2 bands, MCMC, multi-param, time-series |
| 24--25 | Full BioSNICAR pipeline (optional capstone) and decision guide |

**Audience:** Remote sensing practitioners, field scientists, and anyone
working with satellite-derived snow/ice products.  No prior knowledge of
radiative transfer is assumed.


## Requirements

All three notebooks run almost entirely on standard scientific Python.
The final (optional) capstone sections of each notebook use BioSNICAR
itself.

### Core dependencies (all sections except the optional capstone)

```
numpy
matplotlib
scipy
scikit-learn
```

The RT fundamentals notebook also uses:

```
miepython        # Mie scattering calculations
```

The training manual also uses:

```
emcee            # MCMC sampling (Section 19 only)
```

### Optional (for capstone sections)

```
biosnicar        # pip install -e /path/to/biosnicar-py
```

If BioSNICAR is not installed the capstone sections are skipped
automatically.


## Running the notebooks

```bash
# From the repository root
cd notebooks

# Install dependencies (if not already present)
pip install numpy matplotlib scipy scikit-learn miepython emcee

# Launch Jupyter
jupyter notebook
```

Select any notebook and run all cells.  Every section is independent
of BioSNICAR — only the final capstone sections (marked "Optional")
require the package to be installed.

To run the capstone sections, install BioSNICAR in development mode
from the repository root:

```bash
pip install -e .
```

### Expected run times

| Notebook | Without capstone | With capstone |
|----------|-----------------|---------------|
| RT Fundamentals | ~1 minute | ~2 minutes |
| Training Manual | ~2 minutes | ~5 minutes |
| Remote Sensing  | ~1 minute | ~2 minutes |

Times are approximate for a modern laptop.  The training manual capstone
builds an emulator from 500 forward-model evaluations, which accounts
for most of the additional time.
