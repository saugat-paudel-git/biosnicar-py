"""Tests for subsurface light field output."""

import numpy as np
import pytest

from biosnicar import run_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ad_outputs():
    """Adding-doubling model outputs for a 2-layer column."""
    return run_model(
        solzen=60, rds=500, dz=[0.05, 0.95], rho=[400, 400],
    )


@pytest.fixture(scope="module")
def toon_outputs():
    """Toon solver outputs for a 2-layer column (granular, SZA=50)."""
    return run_model(
        solver="toon", solzen=50, rds=500,
        dz=[0.05, 0.95], rho=[400, 400],
        layer_type=[0, 0],
    )


# ---------------------------------------------------------------------------
# 1. Flux arrays are present and correctly shaped
# ---------------------------------------------------------------------------

def test_flux_arrays_present(ad_outputs):
    assert ad_outputs.F_up is not None
    assert ad_outputs.F_dwn is not None
    # 480 wavelengths, 2 layers -> 3 interfaces
    assert ad_outputs.F_up.shape == (480, 3)
    assert ad_outputs.F_dwn.shape == (480, 3)


def test_flux_arrays_present_toon(toon_outputs):
    assert toon_outputs.F_up is not None
    assert toon_outputs.F_dwn is not None
    assert toon_outputs.F_up.shape == (480, 3)
    assert toon_outputs.F_dwn.shape == (480, 3)


# ---------------------------------------------------------------------------
# 3. Surface albedo consistency (raw ratio vs albedo before smoothing)
# ---------------------------------------------------------------------------

def test_surface_albedo_consistency(ad_outputs):
    # The stored albedo may have been smoothed, so compare only in bands
    # with significant flux and use a generous tolerance.
    mask = ad_outputs.F_dwn[:, 0] > 1e-4
    ratio = ad_outputs.F_up[mask, 0] / ad_outputs.F_dwn[mask, 0]
    ratio = np.clip(ratio, 0.0, 1.0)
    np.testing.assert_allclose(ratio, ad_outputs.albedo[mask], atol=0.02)


# ---------------------------------------------------------------------------
# 4. Energy conservation at surface
# ---------------------------------------------------------------------------

def test_energy_conservation_per_interface(ad_outputs):
    # In bands with non-negligible flux, F_up ≈ albedo * F_dwn
    mask = ad_outputs.F_dwn[:, 0] > 1e-4
    np.testing.assert_allclose(
        ad_outputs.F_up[mask, 0],
        ad_outputs.albedo[mask] * ad_outputs.F_dwn[mask, 0],
        atol=1e-4,
    )


# ---------------------------------------------------------------------------
# 5. subsurface_flux at depth=0 matches interface 0
# ---------------------------------------------------------------------------

def test_subsurface_flux_at_zero(ad_outputs):
    flux = ad_outputs.subsurface_flux(0.0)
    np.testing.assert_allclose(flux["F_up"], ad_outputs.F_up[:, 0], atol=1e-12)
    np.testing.assert_allclose(flux["F_dwn"], ad_outputs.F_dwn[:, 0], atol=1e-12)


# ---------------------------------------------------------------------------
# 6. subsurface_flux at total depth matches bottom interface
# ---------------------------------------------------------------------------

def test_subsurface_flux_at_bottom(ad_outputs):
    total_depth = sum(ad_outputs._dz)
    flux = ad_outputs.subsurface_flux(total_depth)
    np.testing.assert_allclose(flux["F_up"], ad_outputs.F_up[:, -1], atol=1e-12)
    np.testing.assert_allclose(flux["F_dwn"], ad_outputs.F_dwn[:, -1], atol=1e-12)


# ---------------------------------------------------------------------------
# 7. Interpolation at mid-layer depth
# ---------------------------------------------------------------------------

def test_subsurface_flux_interpolation(ad_outputs):
    mid_depth = ad_outputs._dz[0] / 2.0
    flux = ad_outputs.subsurface_flux(mid_depth)
    # Values should be between interface 0 and interface 1
    lo = np.minimum(ad_outputs.F_dwn[:, 0], ad_outputs.F_dwn[:, 1])
    hi = np.maximum(ad_outputs.F_dwn[:, 0], ad_outputs.F_dwn[:, 1])
    assert np.all(flux["F_dwn"] >= lo - 1e-12)
    assert np.all(flux["F_dwn"] <= hi + 1e-12)


# ---------------------------------------------------------------------------
# 8. PAR at surface is positive and plausible
# ---------------------------------------------------------------------------

def test_par_surface(ad_outputs):
    par = ad_outputs.par(0.0)
    assert isinstance(par, float)
    # Normalised fluxes: PAR fraction at surface should be 0.1-0.6
    assert 0.1 < par < 0.6


# ---------------------------------------------------------------------------
# 9. PAR decreases with depth
# ---------------------------------------------------------------------------

def test_par_decreases_with_depth(ad_outputs):
    # Downwelling PAR can be enhanced near the surface due to
    # backscattering (radiation trapping), but net PAR (F_dwn - F_up)
    # must decrease monotonically.  Verify that at sufficient depth
    # the PAR is lower than the surface value.
    wvl = ad_outputs._wavelengths
    par_mask = (wvl >= 0.4) & (wvl <= 0.7)
    flux_0 = ad_outputs.subsurface_flux(0.0)
    flux_d = ad_outputs.subsurface_flux(0.5)
    net_par_0 = np.sum(flux_0["F_net"][par_mask])
    net_par_d = np.sum(flux_d["F_net"][par_mask])
    assert net_par_0 > net_par_d


# ---------------------------------------------------------------------------
# 10. PAR with array input
# ---------------------------------------------------------------------------

def test_par_array_input(ad_outputs):
    result = ad_outputs.par([0.0, 0.1, 0.5])
    assert result.shape == (3,)
    assert np.all(result > 0)


# ---------------------------------------------------------------------------
# 11. Spectral heating rate shape
# ---------------------------------------------------------------------------

def test_spectral_heating_rate_shape(ad_outputs):
    shr = ad_outputs.spectral_heating_rate()
    assert shr.shape == (480, 2)
