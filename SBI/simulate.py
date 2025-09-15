import nest

from analyze_simulation import analyze
from utils import plot_vm

from parameters import create_parameters_dict


def simulate(parameters, plot=False):
    import nest

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

    results = analyze(vm, sr, milliseconds)
    if plot:
        plot_vm(vm, save=True)
    return results


if __name__ == "__main__":
    simulate(
        [
            207.8396,
            49.2717,
            -55.4207,
            4.2415,
            19.4002,
            4.7840,
            -46.2139,
            6.1530,
            81.1153,
            117.6044,
        ],
        plot=True,
    )
