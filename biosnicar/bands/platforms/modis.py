"""MODIS (Terra/Aqua): 7 land bands + VIS/NIR/SW broadband, NDSI index.

Uses tophat approximation — bands are wide enough that SRF shape is secondary
at 10 nm model resolution.  Band centres/widths from MODIS Level 1
Calibration documentation.

Band  Centre (µm)  Width (µm)
  1     0.645       0.620–0.670
  2     0.858       0.841–0.876
  3     0.469       0.459–0.479
  4     0.555       0.545–0.565
  5     1.240       1.230–1.250
  6     1.640       1.628–1.652
  7     2.130       2.105–2.155
"""

from biosnicar.bands import BandResult, _register
from biosnicar.bands._core import interval_average

MODIS_BANDS = {
    "B1": (0.620, 0.670),
    "B2": (0.841, 0.876),
    "B3": (0.459, 0.479),
    "B4": (0.545, 0.565),
    "B5": (1.230, 1.250),
    "B6": (1.628, 1.652),
    "B7": (2.105, 2.155),
}

# Broadband aggregations
MODIS_BROAD = {
    "VIS": (0.3, 0.7),
    "NIR": (0.7, 5.0),
    "SW":  (0.3, 5.0),
}


def _modis(albedo, flx_slr):
    r = BandResult("modis")

    for name, (lo, hi) in MODIS_BANDS.items():
        r._set_band(name, interval_average(albedo, flx_slr, lo, hi))

    for name, (lo, hi) in MODIS_BROAD.items():
        r._set_band(name, interval_average(albedo, flx_slr, lo, hi))

    # NDSI = (B4 − B6) / (B4 + B6)
    b4, b6 = r.B4, r.B6
    r._set_index("NDSI", (b4 - b6) / (b4 + b6) if (b4 + b6) != 0 else float("nan"))

    return r


_register("modis", _modis)
