import os
import nest

from analyze_simulation import analyze
from utils import plot_vm, plot_sr
from parameters import create_parameters_dict


def simulate(parameters, results_dir=None):
    try:

        nest.ResetKernel()
        nest.set_verbosity("M_WARNING")

        neuron = nest.Create("aeif_cond_alpha_multisynapse", 1)
        parameters_dict = create_parameters_dict(parameters)
        neuron.set(parameters_dict)
        vm = nest.Create("voltmeter", params={"interval": 0.1})
        sr = nest.Create("spike_recorder")
        nest.Connect(vm, neuron)
        nest.Connect(neuron, sr)

        milliseconds = 5000.0
        nest.Simulate(milliseconds)

        if len(vm.events["V_m"]) == 0:
            print(parameters)

        results = analyze(vm, sr, milliseconds, V_th=parameters_dict["V_th"])
        if results_dir:
            vm_filename = os.path.join(results_dir, f"voltage_trace.png")
            sr_filename = os.path.join(results_dir, f"raster_plot.png")

            plot_vm(vm, save=True, output=vm_filename)
            plot_sr(sr, save=True, output=sr_filename)

        return results
    except nest.kernel.NESTError as e:
        # Handle NEST numerical instabilities and simulation errors
        if (
            "numerical" in str(e).lower()
            or "gsl" in str(e).lower()
            or "integration" in str(e).lower()
        ):
            sim_result = 0, 0, 0, 0
        else:
            raise  # Re-raise other NEST errors
