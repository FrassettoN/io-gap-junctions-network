import nest
import os
from pynestml.frontend.pynestml_frontend import generate_nest_target
from parameters import create_parameters_dict
from utils import plot_vm


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


def initialize_aeif(parameters_dict={}):
    nest.Install("nestml_gap_aeif_cond_exp_neuron_module")
    neurons = nest.Create("aeif_cond_exp_neuron_nestml", 2)
    neurons.set(parameters_dict)
    neurons[0].V_m = neurons[0].V_m - 2

    return neurons


def simulate_network(parameters=[], results_dir=None):
    nest.ResetKernel()
    # generate_code(neuron_model="aeif_cond_alpha_neuron", models_path="../nest_models/")

    nest.resolution = 0.05

    # Change t_ref to refr_t to match NESTML model
    parameters_dict = create_parameters_dict(parameters)
    if "t_ref" in parameters_dict:
        parameters_dict["refr_t"] = parameters_dict.pop("t_ref")
    neurons = initialize_aeif(parameters_dict)

    # GAP CONNECTION
    nest.Connect(
        neurons[0],
        neurons[1],
        {"rule": "one_to_one", "allow_autapses": False, "make_symmetric": True},
        {"synapse_model": "gap_junction", "weight": 0.5},
    )

    # Voltmeter connected to all neurons
    vm = nest.Create("voltmeter", params={"interval": 0.1})
    nest.Connect(vm, neurons, "all_to_all")

    sr = nest.Create("spike_recorder")
    nest.Connect(neurons, sr)

    # Simulation
    simulation_time = 5000.0
    nest.Simulate(simulation_time)

    # Plots
    if results_dir:
        vm_filename = os.path.join(results_dir, f"gap_trace.png")
        plot_vm(vm, save=True, output=vm_filename)


if __name__ == "__main__":
    simulate_network(parameters=[], results_dir="./")
