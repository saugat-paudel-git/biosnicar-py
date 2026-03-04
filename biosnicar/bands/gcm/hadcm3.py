"""HadCM3: 6-band Edwards-Slingo shortwave radiation scheme.

Band boundaries from Edwards & Slingo (1996):
    Band 1: 0.20–0.32 µm  (UV)
    Band 2: 0.32–0.69 µm  (visible)
    Band 3: 0.69–1.19 µm  (near-IR)
    Band 4: 1.19–2.38 µm  (shortwave-IR)
    Band 5: 2.38–4.00 µm  (mid-IR)
    Band 6: 4.00–5.00 µm  (clipped to model range)
"""

from biosnicar.bands import BandResult, _register
from biosnicar.bands._core import interval_average

HADCM3_BANDS = {
    "es1": (0.20, 0.32),
    "es2": (0.32, 0.69),
    "es3": (0.69, 1.19),
    "es4": (1.19, 2.38),
    "es5": (2.38, 4.00),
    "es6": (4.00, 5.00),
}


def _hadcm3(albedo, flx_slr):
    r = BandResult("hadcm3")
    for name, (lo, hi) in HADCM3_BANDS.items():
        r._set_band(name, interval_average(albedo, flx_slr, lo, hi))
    return r


_register("hadcm3", _hadcm3)
