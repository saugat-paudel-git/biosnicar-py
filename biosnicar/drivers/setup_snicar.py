#!/usr/bin/python

from pathlib import Path

from biosnicar.utils.load_inputs import load_inputs
from biosnicar.classes import (
    Ice,
    Illumination,
    Impurity,
    ModelConfig,
    PlotConfig,
    RTConfig,
)


def setup_snicar(input_file):
    """Build all model objects from a YAML configuration file.

    Args:
        input_file: Path to the YAML config, or ``"default"`` to use the
            bundled ``inputs.yaml``.

    Returns:
        Tuple of (ice, illumination, rt_config, model_config, plot_config,
        impurities).
    """
    # define input file
    if input_file == "default":
        BIOSNICAR_SRC_PATH = Path(__file__).resolve().parent
        input_file = BIOSNICAR_SRC_PATH.joinpath("../inputs.yaml").as_posix()

    else:
        input_file = input_file
    impurities = build_impurities_array(input_file)
    (
        ice,
        illumination,
        rt_config,
        model_config,
        plot_config,
    ) = build_classes(input_file)

    return (
        ice,
        illumination,
        rt_config,
        model_config,
        plot_config,
        impurities,
    )


def build_classes(input_file):
    """Instantiate model classes from a YAML config file.

    Args:
        input_file: Path to the YAML configuration file.

    Returns:
        Tuple of (ice, illumination, rt_config, model_config, plot_config).
    """

    ice = Ice(input_file)
    illumination = Illumination(input_file)
    rt_config = RTConfig(input_file)
    model_config = ModelConfig(input_file)
    plot_config = PlotConfig(input_file)

    return ice, illumination, rt_config, model_config, plot_config


def build_impurities_array(input_file):
    """Create a list of Impurity instances from YAML config.

    Args:
        input_file: Path to the YAML configuration file.

    Returns:
        List of :class:`~biosnicar.classes.impurity.Impurity` instances.
    """

    inputs = load_inputs(input_file)

    impurities = []

    for key, entry in inputs["IMPURITIES"].items():
        # New format: YAML key is the name (e.g. "black_carbon")
        # Old format: NAME field holds the name (e.g. "bc")
        name = entry.get("NAME", key)
        file = entry["FILE"]
        coated = entry["COATED"]
        unit = entry["UNIT"]
        conc = entry["CONC"]
        impurities.append(Impurity(file, coated, unit, name, conc))

    return impurities


def get_impurity_names(input_file="default"):
    """Return ordered list of impurity names from the YAML config.

    Args:
        input_file: Path to the YAML config, or ``"default"`` for the
            bundled ``inputs.yaml``.

    Returns:
        List of impurity name strings (e.g. ``["black_carbon",
        "snow_algae", "glacier_algae"]``).
    """
    if input_file == "default":
        input_file = Path(__file__).resolve().parent.joinpath(
            "../inputs.yaml"
        ).as_posix()

    inputs = load_inputs(input_file)
    names = []
    for key, entry in inputs["IMPURITIES"].items():
        names.append(entry.get("NAME", key))
    return names


if __name__ == "__main__":
    pass
