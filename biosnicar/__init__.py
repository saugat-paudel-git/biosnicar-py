from pathlib import Path

__version__ = "2.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def run_model(*args, **kwargs):
    """Run the BioSNICAR forward model. See :func:`biosnicar.drivers.run_model.run_model`."""
    from biosnicar.drivers.run_model import run_model as _run_model

    return _run_model(*args, **kwargs)


def run_emulator(*args, **kwargs):
    """Evaluate a trained emulator. See :func:`biosnicar.drivers.run_emulator.run_emulator`."""
    from biosnicar.drivers.run_emulator import run_emulator as _run_emulator

    return _run_emulator(*args, **kwargs)


def to_platform(*args, **kwargs):
    """Convolve spectral albedo onto platform bands. See :func:`biosnicar.bands.to_platform`."""
    from biosnicar.bands import to_platform as _to_platform

    return _to_platform(*args, **kwargs)
