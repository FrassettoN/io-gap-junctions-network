import os
import nest
from scipy.signal import find_peaks

from analyze_simulation import analyze
from utils import plot_vm, plot_sr

# Configuration constants
SIMULATION_CONFIG = {
    "adex": {
        "module": "nestml_gap_aeif_cond_exp_neuron_module",
        "model": "aeif_cond_exp_neuron_nestml",
    },
    "eglif": {
        "module": "nestml_gap_eglif_multirec_opt_module",
        "model": "eglif_multirec_opt_nestml",
    },
}


def _run_simulation(model_type, parameters, results_dir=None):
    """Core simulation logic shared between models."""
    # Import model-specific function
    if model_type == "adex":
        from adex import create_parameters_dict
    elif model_type == "eglif":
        from eglif import create_parameters_dict
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Setup NEST
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")

    # Get model configuration
    config = SIMULATION_CONFIG[model_type]

    # Create and configure neuron
    nest.Install(config["module"])
    neuron = nest.Create(config["model"], 1)
    parameters_dict = create_parameters_dict(parameters)
    neuron.set(parameters_dict)

    # Create recording devices
    vm = nest.Create("voltmeter", params={"interval": 0.1})
    sr = nest.Create("spike_recorder")
    nest.Connect(vm, neuron)
    nest.Connect(neuron, sr)

    # start = 800
    # duration = 200
    # amplitude = -5
    # dc = nest.Create(
    #     "dc_generator",
    #     params={"amplitude": amplitude, "start": start, "stop": start + duration},
    # )
    # nest.Connect(dc, neuron, syn_spec={"weight": 1.0})

    # Run simulation
    simulation_time = 5000.0
    nest.Simulate(simulation_time)

    # Analyze results
    results = analyze(vm, sr, simulation_time)

    # Save plots if directory provided and results exist
    if results_dir and results:
        _save_simulation_plots(vm, sr, results_dir, model_type)

    return results


def simulate_adex(parameters, results_dir=None):
    """Simulate AdEx neuron model."""
    try:
        return _run_simulation("adex", parameters, results_dir)
    except nest.kernel.NESTError as e:
        return _handle_nest_error(e)


def simulate_eglif(parameters, results_dir=None):
    """Simulate EGLIF neuron model."""
    try:
        return _run_simulation("eglif", parameters, results_dir)
    except nest.kernel.NESTError as e:
        return _handle_nest_error(e)


def simulate(parameters, results_dir=None, model="adex"):
    """Generic simulation function."""
    if model == "adex":
        return simulate_adex(parameters, results_dir)
    elif model == "eglif":
        return simulate_eglif(parameters, results_dir)
    else:
        raise ValueError(f"Unknown model: {model}")


def _save_simulation_plots(vm, sr, results_dir, model_name):
    """Save voltage and spike plots to results directory."""
    try:
        vm_filename = os.path.join(results_dir, f"voltage_trace.png")
        sr_filename = os.path.join(results_dir, f"raster_plot.png")

        plot_vm(vm, save=True, output=vm_filename)
        plot_sr(sr, save=True, output=sr_filename)
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")


def _handle_nest_error(e):
    """Handle NEST simulation errors gracefully."""
    error_msg = str(e).lower()
    if any(keyword in error_msg for keyword in ["numerical", "gsl", "integration"]):
        print(f"Numerical instability detected: {e}")
        return (0, 0, 0, 0, 0)  # Return default values
    else:
        print(f"NEST error: {e}")
        raise  # Re-raise non-numerical errors


if __name__ == "__main__":
    parameters = [
        2.6103e02,
        1.0533e01,
        1.6592e01,
        2.6531e00,
        5.1926e-01,
        9.4476e-02,
        -3.3671e01,
        9.9121e-01,
        7.3815e-01,
        1.6394e03,
        1.9759e03,
    ]
    simulate(
        parameters,
        "./",
        model="eglif",
    )
    # results = simulate(parameters, "./", model="adex")  # Use generic function
    # print(results)
