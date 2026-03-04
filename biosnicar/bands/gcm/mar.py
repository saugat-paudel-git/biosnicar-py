"""MAR regional climate model: 4-band shortwave scheme.

Band boundaries from Fettweis et al. / MAR documentation:
    SW1: 0.25–0.69 µm
    SW2: 0.69–1.19 µm
    SW3: 1.19–2.38 µm
    SW4: 2.38–4.00 µm
"""

from biosnicar.bands import BandResult, _register
from biosnicar.bands._core import interval_average

MAR_BANDS = {
    "sw1": (0.25, 0.69),
    "sw2": (0.69, 1.19),
    "sw3": (1.19, 2.38),
    "sw4": (2.38, 4.00),
}


def _mar(albedo, flx_slr):
    r = BandResult("mar")
    for name, (lo, hi) in MAR_BANDS.items():
        r._set_band(name, interval_average(albedo, flx_slr, lo, hi))
    return r


_register("mar", _mar)
