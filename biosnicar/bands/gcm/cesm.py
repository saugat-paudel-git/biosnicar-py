"""CESM band definitions: 2-band (VIS/NIR) and 14-band RRTMG.

CESM 2-band
    VIS: 0.2–0.7 µm, NIR: 0.7–5.0 µm

CESM RRTMG (Iacono et al. 2008, Table 1 shortwave bands)
    14 bands spanning 0.2–12.2 µm (only bands within the BioSNICAR
    wavelength range contribute).
"""

from biosnicar.bands import BandResult, _register
from biosnicar.bands._core import interval_average

# ── CESM 2-band ─────────────────────────────────────────────────────
CESM_2BAND = {
    "vis": (0.2, 0.7),
    "nir": (0.7, 5.0),
}


def _cesm2band(albedo, flx_slr):
    r = BandResult("cesm2band")
    for name, (lo, hi) in CESM_2BAND.items():
        r._set_band(name, interval_average(albedo, flx_slr, lo, hi))
    return r


_register("cesm2band", _cesm2band)

# ── CESM RRTMG shortwave (Iacono et al. 2008) ──────────────────────
# Bands are numbered 1-14; wavelength bounds in µm.
# Only bands overlapping the 0.205–4.995 µm model grid are useful.
CESM_RRTMG = {
    "sw1":  (3.077, 3.846),
    "sw2":  (2.500, 3.077),
    "sw3":  (2.151, 2.500),
    "sw4":  (1.942, 2.151),
    "sw5":  (1.626, 1.942),
    "sw6":  (1.299, 1.626),
    "sw7":  (1.242, 1.299),
    "sw8":  (0.778, 1.242),
    "sw9":  (0.625, 0.778),
    "sw10": (0.442, 0.625),
    "sw11": (0.345, 0.442),
    "sw12": (0.263, 0.345),
    "sw13": (0.200, 0.263),
    "sw14": (3.846, 12.195),  # mostly beyond model range
}


def _cesmrrtmg(albedo, flx_slr):
    r = BandResult("cesmrrtmg")
    for name, (lo, hi) in CESM_RRTMG.items():
        r._set_band(name, interval_average(albedo, flx_slr, lo, hi))
    return r


_register("cesmrrtmg", _cesmrrtmg)
