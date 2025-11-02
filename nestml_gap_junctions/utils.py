import os
from pynestml.frontend.pynestml_frontend import generate_nest_target


def generate_code(neuron_model: str, models_path=""):
    """
    Generate NEST code for neuron model with gap junction support.
    Parameters
    ----------
    neuron_model : str
        Name of the neuron model to compile. This should correspond to a
        .nestml file containing the neuron model definition.
    models_path : str, optional
        Path to the directory containing the NESTML model files.
        Default is empty string (current directory).
    """
    codegen_opts = {
        "gap_junctions": {
            "enable": True,
            "gap_current_port": "I_stim",
            "membrane_potential_variable": "V_m",
        }
    }

    files = os.path.join(models_path, neuron_model + ".nestml")
    generate_nest_target(
        input_path=files,
        logging_level="WARNING",
        module_name="nestml_gap_" + neuron_model + "_module",
        suffix="_nestml",
        codegen_opts=codegen_opts,
    )

    return neuron_model
